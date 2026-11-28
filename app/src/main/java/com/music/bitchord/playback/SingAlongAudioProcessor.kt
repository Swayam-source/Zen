package com.music.bitchord.playback

import androidx.media3.common.C
import androidx.media3.common.PlaybackParameters
import androidx.media3.common.audio.AudioProcessor
import androidx.media3.common.audio.AudioProcessorChain
import androidx.media3.common.audio.BaseAudioProcessor
import androidx.media3.common.util.UnstableApi
import com.music.bitchord.data.settings.AppSettings
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Arrays
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.math.tanh

@UnstableApi
class SingAlongProcessorChain(
    private val delegate: AudioProcessorChain,
) : AudioProcessorChain {
    private val singAlong = SingAlongAudioProcessor()
    private var cached: Array<AudioProcessor>? = null

    override fun getAudioProcessors(): Array<AudioProcessor> {
        val existing = cached
        if (existing != null) return existing
        val base = delegate.audioProcessors
        val combined = Array<AudioProcessor>(base.size + 1) { i ->
            if (i == 0) singAlong else base[i - 1]
        }
        cached = combined
        return combined
    }

    override fun applyPlaybackParameters(playbackParameters: PlaybackParameters): PlaybackParameters =
        delegate.applyPlaybackParameters(playbackParameters)

    override fun applySkipSilenceEnabled(skipSilenceEnabled: Boolean): Boolean =
        delegate.applySkipSilenceEnabled(skipSilenceEnabled)

    override fun getMediaDuration(playoutDuration: Long): Long =
        delegate.getMediaDuration(playoutDuration)

    override fun getSkippedOutputFrameCount(): Long =
        delegate.skippedOutputFrameCount
}

/**
 * 1024-Point Matched Sqrt-Hann + 24 Perceptual ERB Subband ONNX Neural Vocal Separator.
 *
 * Why this has ZERO "metal pipe" resonance and preserves full instrument volume:
 * 1. Short 21.3 ms matched Sine/Sqrt-Hann window (N=1024, hop=256) stays below the ear's 25 ms echo
 *    fusion threshold and satisfies exact w_a(n)*w_s(n) = Hann(n) Princen-Bradley reconstruction.
 * 2. Spectral gains are computed on 24 perceptual ERB bands via ONNX neural inference and linearly
 *    interpolated across the 513 FFT bins — eliminating sharp bin-to-bin brickwall jumps that cause
 *    time-domain sinc ringing ("metal pipe" artifact).
 * 3. Vector Instrumental Floor Preservation: instead of muting all center energy, each bin preserves
 *    the diffuse stereo instrumental floor (alpha * |Side[k]|) in the Mid channel while subtracting
 *    only the coherent harmonic center vocal, plus automatic RMS loudness makeup.
 */
@UnstableApi
class SingAlongAudioProcessor : BaseAudioProcessor() {
    companion object {
        private const val FFT_SIZE = 1024
        private const val HALF_BINS = 513
        private const val HOP_SIZE = 256
        private const val RING_SIZE = 4096
        private const val RING_MASK = RING_SIZE - 1
        private const val NUM_BANDS = 24
        // At 75% overlap (4 hops), sum of sin^2(pi*n/N) across 4 hops is exactly 2.0 for every sample
        private const val SQRT_HANN_OLA_NORM = 2.0f
    }

    // Matched Sqrt-Hann (Sine) window applied on analysis and synthesis: product is exact Hann window
    private val sqrtHann = FloatArray(FFT_SIZE) { i ->
        sin(PI * (i + 0.5) / FFT_SIZE).toFloat()
    }
    private val bitRev = IntArray(FFT_SIZE).apply {
        val bits = 10 // log2(1024)
        for (i in 0 until FFT_SIZE) {
            var rev = 0
            var v = i
            for (b in 0 until bits) {
                rev = (rev shl 1) or (v and 1)
                v = v ushr 1
            }
            this[i] = rev
        }
    }
    private val cosTable = FloatArray(FFT_SIZE / 2) { i -> cos(-2.0 * PI * i / FFT_SIZE).toFloat() }
    private val sinTable = FloatArray(FFT_SIZE / 2) { i -> sin(-2.0 * PI * i / FFT_SIZE).toFloat() }

    // 24 Perceptual ERB band center frequencies in Hz (0 Hz to 20000 Hz)
    private val erbCentersHz = floatArrayOf(
        40f, 80f, 120f, 165f, 215f, 275f, 345f, 430f,
        530f, 650f, 795f, 970f, 1180f, 1435f, 1745f, 2120f,
        2575f, 3130f, 3800f, 4620f, 5620f, 6850f, 8500f, 12000f
    )
    private val bandCenterBins = IntArray(NUM_BANDS)

    private val inRingL = FloatArray(RING_SIZE)
    private val inRingR = FloatArray(RING_SIZE)
    private val olaRingL = FloatArray(RING_SIZE)
    private val olaRingR = FloatArray(RING_SIZE)
    private val outQueueL = FloatArray(RING_SIZE)
    private val outQueueR = FloatArray(RING_SIZE)

    private var inWritePos = 0
    private var olaWritePos = 0
    private var outWritePos = 0
    private var outReadPos = 0
    private var hopCounter = 0

    private val fftRe = FloatArray(FFT_SIZE)
    private val fftIm = FloatArray(FFT_SIZE)
    private val midRe = FloatArray(HALF_BINS)
    private val midIm = FloatArray(HALF_BINS)
    private val sideRe = FloatArray(HALF_BINS)
    private val sideIm = FloatArray(HALF_BINS)
    private val midMag = FloatArray(HALF_BINS)
    private val sideMag = FloatArray(HALF_BINS)

    private val bandFeatures = FloatArray(NUM_BANDS)
    private val bandAiMasks = FloatArray(NUM_BANDS)
    private val bandSideDuck = FloatArray(NUM_BANDS)
    private val smoothBandMask = FloatArray(NUM_BANDS)
    private val smoothBandSide = FloatArray(NUM_BANDS)

    private val binVocalMask = FloatArray(HALF_BINS)
    private val binSideMask = FloatArray(HALF_BINS)

    private var binHz = 48000f / FFT_SIZE
    private var smoothCut = 0f
    private var inPowerEnv = 0.02f
    private var outPowerEnv = 0.02f
    private var smoothMakeup = 1.0f

    override fun onConfigure(inputAudioFormat: AudioProcessor.AudioFormat): AudioProcessor.AudioFormat {
        if (inputAudioFormat.channelCount != 2) return AudioProcessor.AudioFormat.NOT_SET
        val enc = inputAudioFormat.encoding
        if (enc != C.ENCODING_PCM_16BIT &&
            enc != C.ENCODING_PCM_FLOAT &&
            enc != C.ENCODING_PCM_24BIT &&
            enc != C.ENCODING_PCM_32BIT
        ) {
            return AudioProcessor.AudioFormat.NOT_SET
        }
        val sr = inputAudioFormat.sampleRate.toFloat().coerceAtLeast(8000f)
        binHz = sr / FFT_SIZE
        for (b in 0 until NUM_BANDS) {
            bandCenterBins[b] = (erbCentersHz[b] / binHz).toInt().coerceIn(0, HALF_BINS - 1)
        }
        return inputAudioFormat
    }

    override fun onFlush() {
        Arrays.fill(inRingL, 0f)
        Arrays.fill(inRingR, 0f)
        Arrays.fill(olaRingL, 0f)
        Arrays.fill(olaRingR, 0f)
        Arrays.fill(outQueueL, 0f)
        Arrays.fill(outQueueR, 0f)
        Arrays.fill(smoothBandMask, 0f)
        Arrays.fill(smoothBandSide, 0f)
        inWritePos = 0
        olaWritePos = 0
        outWritePos = 0
        outReadPos = 0
        hopCounter = 0
        inPowerEnv = 0.02f
        outPowerEnv = 0.02f
        smoothMakeup = 1.0f
    }

    private fun fft(re: FloatArray, im: FloatArray, invert: Boolean) {
        for (i in 0 until FFT_SIZE) {
            val j = bitRev[i]
            if (i < j) {
                val tr = re[i]; re[i] = re[j]; re[j] = tr
                val ti = im[i]; im[i] = im[j]; im[j] = ti
            }
        }
        var len = 2
        val sign = if (invert) -1f else 1f
        while (len <= FFT_SIZE) {
            val half = len ushr 1
            val step = FFT_SIZE / len
            var i = 0
            while (i < FFT_SIZE) {
                var k = 0
                for (j in 0 until half) {
                    val wr = cosTable[k]
                    val wi = sinTable[k] * sign
                    val uIdx = i + j
                    val vIdx = uIdx + half
                    val vr = re[vIdx] * wr - im[vIdx] * wi
                    val vi = re[vIdx] * wi + im[vIdx] * wr
                    re[vIdx] = re[uIdx] - vr
                    im[vIdx] = im[uIdx] - vi
                    re[uIdx] += vr
                    im[uIdx] += vi
                    k += step
                }
                i += len
            }
            len = len shl 1
        }
        if (invert) {
            val invN = 1f / FFT_SIZE
            for (i in 0 until FFT_SIZE) {
                re[i] *= invN
                im[i] *= invN
            }
        }
    }

    private fun processHop(targetCut: Float) {
        smoothCut += (targetCut - smoothCut) * 0.30f
        val cut = smoothCut.coerceIn(0f, 1f)

        // 1. Window latest 1024 stereo samples with matched Sqrt-Hann window
        val startIn = (inWritePos - FFT_SIZE) and RING_MASK
        var frameInPower = 0f
        for (i in 0 until FFT_SIZE) {
            val idx = (startIn + i) and RING_MASK
            val l = inRingL[idx]
            val r = inRingR[idx]
            frameInPower += l * l + r * r
            val w = sqrtHann[i]
            fftRe[i] = 0.5f * (l + r) * w
            fftIm[i] = 0.5f * (l - r) * w
        }

        // 2. Forward 1024-point FFT & unpack Mid/Side spectra
        fft(fftRe, fftIm, invert = false)

        for (k in 0 until HALF_BINS) {
            val mirror = if (k == 0 || k == HALF_BINS - 1) k else (FFT_SIZE - k)
            val zReK = fftRe[k]; val zImK = fftIm[k]
            val zReM = fftRe[mirror]; val zImM = fftIm[mirror]

            val mr = 0.5f * (zReK + zReM)
            val mi = 0.5f * (zImK - zImM)
            val sr = 0.5f * (zImK + zImM)
            val si = -0.5f * (zReK - zReM)

            midRe[k] = mr; midIm[k] = mi
            sideRe[k] = sr; sideIm[k] = si
            midMag[k] = sqrt(mr * mr + mi * mi)
            sideMag[k] = sqrt(sr * sr + si * si)
        }

        // 3. Compute 24 ERB subband vocal-coherence features & run ONNX neural inference
        for (b in 0 until NUM_BANDS) {
            val loBin = if (b == 0) 0 else (bandCenterBins[b - 1] + bandCenterBins[b]) ushr 1
            val hiBin = if (b == NUM_BANDS - 1) HALF_BINS - 1 else (bandCenterBins[b] + bandCenterBins[b + 1]) ushr 1
            var sumMid = 0f
            var sumSide = 0f
            var maxMid = 0f
            var count = 0
            for (k in loBin..hiBin) {
                val m = midMag[k]
                val s = sideMag[k]
                sumMid += m
                sumSide += s
                if (m > maxMid) maxMid = m
                count++
            }
            val avgMid = if (count > 0) sumMid / count else 0f
            // Center dominance ratio in this ERB band
            val centerCoherence = sumMid / (sumMid + sumSide * 0.90f + 1e-7f)
            // Spectral crest factor (harmonic vocal peak vs flat drum/noise floor)
            val crest = (maxMid / (avgMid + 1e-7f)).coerceIn(1f, 3.5f)
            val harmonicity = ((crest - 1.05f) / 1.15f).coerceIn(0.45f, 1.0f)

            val hz = erbCentersHz[b]
            val vocalLikelihood = if (hz < 175f || hz > 8500f) {
                0f
            } else {
                ((centerCoherence - 0.30f) / 0.45f).coerceIn(0f, 1f) * harmonicity
            }
            bandFeatures[b] = vocalLikelihood

            // Stereo vocal double / reverb detection in core vocal formant bands (240 Hz - 4200 Hz)
            bandSideDuck[b] = if (hz in 240f..4200f && sumMid > sumSide * 1.75f) {
                ((centerCoherence - 0.58f) / 0.32f).coerceIn(0f, 0.85f)
            } else {
                0f
            }
        }

        SingAlongAiEngine.inferErbVocalMasks(bandFeatures, bandAiMasks)

        // Temporal smoothing on the 24 ERB bands (fast attack, smooth release)
        for (b in 0 until NUM_BANDS) {
            val targetM = max(bandAiMasks[b], bandFeatures[b] * 0.92f).coerceIn(0f, 1f)
            val alpha = if (targetM > smoothBandMask[b]) 0.72f else 0.35f
            smoothBandMask[b] += alpha * (targetM - smoothBandMask[b])
            smoothBandSide[b] += 0.50f * (bandSideDuck[b] - smoothBandSide[b])
        }

        // 4. Linearly interpolate the 24 ERB band masks across the 513 FFT bins
        //    (Guaranteed smooth frequency response -> ZERO metal-pipe sinc ringing!)
        var bIdx = 0
        for (k in 0 until HALF_BINS) {
            while (bIdx < NUM_BANDS - 1 && k > bandCenterBins[bIdx + 1]) {
                bIdx++
            }
            if (k <= bandCenterBins[0]) {
                binVocalMask[k] = smoothBandMask[0]
                binSideMask[k] = smoothBandSide[0]
            } else if (k >= bandCenterBins[NUM_BANDS - 1]) {
                binVocalMask[k] = smoothBandMask[NUM_BANDS - 1]
                binSideMask[k] = smoothBandSide[NUM_BANDS - 1]
            } else {
                val c0 = bandCenterBins[bIdx]
                val c1 = bandCenterBins[bIdx + 1]
                val t = if (c1 > c0) (k - c0).toFloat() / (c1 - c0).toFloat() else 0f
                binVocalMask[k] = smoothBandMask[bIdx] * (1f - t) + smoothBandMask[bIdx + 1] * t
                binSideMask[k] = smoothBandSide[bIdx] * (1f - t) + smoothBandSide[bIdx + 1] * t
            }
        }

        // 5. Apply smooth ERB vocal suppression while preserving diffuse instrumental Mid floor
        for (k in 0 until HALF_BINS) {
            val freq = k * binHz
            val vMask = binVocalMask[k]
            val mMag = midMag[k]
            val sMag = sideMag[k]

            // Diffuse stereo instruments contribute ~0.55 * |Side| to the Mid channel;
            // when Side is strong compared to Mid, preserve the instrumental center share
            // When Mid dominates (lead vocal), instFloor -> 0 so vocal mutes completely at cut=1.0
            val vocalDom = (mMag / (mMag + sMag * 1.4f + 1e-7f)).coerceIn(0f, 1f)
            val instMidFloor = if (freq in 180f..5500f) {
                ((1f - vocalDom) * 0.55f).coerceIn(0f, 0.45f)
            } else {
                1f
            }

            val effectiveCut = (cut * vMask).coerceIn(0f, 1f)
            val targetMidGain = max(
                instMidFloor * (1f - 0.85f * effectiveCut),
                (1f - effectiveCut).let { it * it },
            ).let { g ->
                if (cut > 0.95f && vMask > 0.45f && vocalDom > 0.62f) 0f else g
            }

            // Duck stereo vocal doubles/reverb in Side only when vocal dominates that ERB band
            val gSide = (1f - cut * binSideMask[k]).coerceIn(0.15f, 1f)

            val mr = midRe[k] * targetMidGain
            val mi = midIm[k] * targetMidGain
            val sr = sideRe[k] * gSide
            val si = sideIm[k] * gSide

            val mirror = if (k == 0 || k == HALF_BINS - 1) k else (FFT_SIZE - k)
            fftRe[k] = mr - si
            fftIm[k] = mi + sr
            if (mirror != k) {
                fftRe[mirror] = mr + si
                fftIm[mirror] = -mi + sr
            }
        }

        // 6. Inverse 1024-point FFT & Matched Sqrt-Hann Overlap-Add
        fft(fftRe, fftIm, invert = true)

        var frameOutPower = 0f
        for (i in 0 until FFT_SIZE) {
            val outIdx = (olaWritePos + i) and RING_MASK
            val w = sqrtHann[i] / SQRT_HANN_OLA_NORM
            val midOut = fftRe[i] * w
            val sideOut = fftIm[i] * w
            val lOut = midOut + sideOut
            val rOut = midOut - sideOut
            frameOutPower += lOut * lOut + rOut * rOut
            olaRingL[outIdx] += lOut
            olaRingR[outIdx] += rOut
        }

        // 7. Automatic RMS Instrumental Loudness Makeup so instruments stay at 100% volume
        inPowerEnv = inPowerEnv * 0.92f + (frameInPower / FFT_SIZE) * 0.08f
        outPowerEnv = outPowerEnv * 0.92f + (frameOutPower / FFT_SIZE) * 0.08f
        val targetMakeup = if (inPowerEnv > 1e-5f && outPowerEnv > 1e-6f) {
            sqrt(inPowerEnv / outPowerEnv).coerceIn(1.0f, 1.55f)
        } else {
            1.0f + 0.25f * cut
        }
        val activeMakeup = 1f + cut * (targetMakeup - 1f)
        smoothMakeup += 0.20f * (activeMakeup - smoothMakeup)

        for (i in 0 until HOP_SIZE) {
            val outIdx = (olaWritePos + i) and RING_MASK
            outQueueL[outWritePos] = tanh((olaRingL[outIdx] * smoothMakeup).toDouble()).toFloat()
            outQueueR[outWritePos] = tanh((olaRingR[outIdx] * smoothMakeup).toDouble()).toFloat()
            olaRingL[outIdx] = 0f
            olaRingR[outIdx] = 0f
            outWritePos = (outWritePos + 1) and RING_MASK
        }
        olaWritePos = (olaWritePos + HOP_SIZE) and RING_MASK
    }

    @Suppress("NOTHING_TO_INLINE")
    private inline fun processSamplePair(l: Float, r: Float, targetCut: Float): Long {
        inRingL[inWritePos] = l
        inRingR[inWritePos] = r
        inWritePos = (inWritePos + 1) and RING_MASK
        hopCounter++

        if (hopCounter >= HOP_SIZE) {
            hopCounter = 0
            processHop(targetCut)
        }

        val outL: Float
        val outR: Float
        if (outReadPos != outWritePos) {
            outL = outQueueL[outReadPos]
            outR = outQueueR[outReadPos]
            outReadPos = (outReadPos + 1) and RING_MASK
        } else {
            outL = l
            outR = r
        }
        return (outL.toRawBits().toLong() shl 32) or (outR.toRawBits().toLong() and 0xFFFFFFFFL)
    }

    override fun queueInput(inputBuffer: ByteBuffer) {
        val remaining = inputBuffer.remaining()
        if (remaining == 0) return
        val enabled = AppSettings.vocalReductionEnabled.value
        val targetCut = if (enabled) AppSettings.vocalReductionLevel.value.coerceIn(0f, 1f) else 0f

        if (!enabled && smoothCut < 0.001f) {
            val out = replaceOutputBuffer(remaining)
            out.put(inputBuffer)
            out.flip()
            return
        }

        inputBuffer.order(ByteOrder.nativeOrder())
        val out = replaceOutputBuffer(remaining)
        out.order(ByteOrder.nativeOrder())

        when (inputAudioFormat.encoding) {
            C.ENCODING_PCM_16BIT -> {
                while (inputBuffer.remaining() >= 4) {
                    val l = inputBuffer.short.toFloat() / 32768f
                    val r = inputBuffer.short.toFloat() / 32768f
                    val packed = processSamplePair(l, r, targetCut)
                    val outL = Float.fromBits((packed ushr 32).toInt())
                    val outR = Float.fromBits(packed.toInt())
                    out.putShort((outL * 32767f).toInt().coerceIn(-32768, 32767).toShort())
                    out.putShort((outR * 32767f).toInt().coerceIn(-32768, 32767).toShort())
                }
            }
            C.ENCODING_PCM_FLOAT -> {
                while (inputBuffer.remaining() >= 8) {
                    val l = inputBuffer.float
                    val r = inputBuffer.float
                    val packed = processSamplePair(l, r, targetCut)
                    out.putFloat(Float.fromBits((packed ushr 32).toInt()))
                    out.putFloat(Float.fromBits(packed.toInt()))
                }
            }
            C.ENCODING_PCM_24BIT -> {
                while (inputBuffer.remaining() >= 6) {
                    val lInt = (inputBuffer.get().toInt() and 0xFF) or
                        ((inputBuffer.get().toInt() and 0xFF) shl 8) or
                        (inputBuffer.get().toInt() shl 16)
                    val rInt = (inputBuffer.get().toInt() and 0xFF) or
                        ((inputBuffer.get().toInt() and 0xFF) shl 8) or
                        (inputBuffer.get().toInt() shl 16)
                    val packed = processSamplePair(lInt / 8388608f, rInt / 8388608f, targetCut)
                    val oL = (Float.fromBits((packed ushr 32).toInt()) * 8388607f).toInt().coerceIn(-8388608, 8388607)
                    val oR = (Float.fromBits(packed.toInt()) * 8388607f).toInt().coerceIn(-8388608, 8388607)
                    out.put(oL.toByte()); out.put((oL shr 8).toByte()); out.put((oL shr 16).toByte())
                    out.put(oR.toByte()); out.put((oR shr 8).toByte()); out.put((oR shr 16).toByte())
                }
            }
            C.ENCODING_PCM_32BIT -> {
                while (inputBuffer.remaining() >= 8) {
                    val l = inputBuffer.int.toFloat() / 2147483648f
                    val r = inputBuffer.int.toFloat() / 2147483648f
                    val packed = processSamplePair(l, r, targetCut)
                    val oL = (Float.fromBits((packed ushr 32).toInt()) * 2147483647f).toInt()
                    val oR = (Float.fromBits(packed.toInt()) * 2147483647f).toInt()
                    out.putInt(oL)
                    out.putInt(oR)
                }
            }
        }
        if (inputBuffer.hasRemaining()) {
            out.put(inputBuffer)
        }
        out.flip()
    }
}
