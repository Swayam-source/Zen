package com.music.bitchord.playback

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import java.io.File
import java.nio.FloatBuffer

/**
 * Built-in ONNX Runtime Neural Vocal Mask Engine.
 * Loads the pre-packaged sing_along_vocal_net.onnx directly from APK assets with zero popups
 * or network downloads.
 */
object SingAlongAiEngine {
    const val NUM_ERB_BANDS = 24
    private const val ASSET_MODEL_NAME = "sing_along_vocal_net.onnx"

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null

    @Volatile var modelReady: Boolean = false
        private set

    fun init(context: Context) {
        val appCtx = context.applicationContext
        scope.launch(Dispatchers.IO) {
            runCatching {
                val outFile = File(appCtx.filesDir, ASSET_MODEL_NAME)
                if (!outFile.exists() || outFile.length() == 0L) {
                    appCtx.assets.open(ASSET_MODEL_NAME).use { input ->
                        outFile.outputStream().use { output -> input.copyTo(output) }
                    }
                }
                val env = OrtEnvironment.getEnvironment()
                val opts = OrtSession.SessionOptions().apply {
                    setIntraOpNumThreads(2)
                    setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                }
                ortEnv = env
                ortSession = env.createSession(outFile.absolutePath, opts)
                modelReady = true
            }
        }
    }

    /**
     * Evaluates the ONNX neural network on the 24 ERB band vocal-coherence features synchronously
     * in microseconds, writing the 24 smooth vocal suppression probabilities into [outMasks].
     */
    fun inferErbVocalMasks(inFeatures: FloatArray, outMasks: FloatArray) {
        val session = ortSession
        val env = ortEnv
        if (!modelReady || session == null || env == null) {
            System.arraycopy(inFeatures, 0, outMasks, 0, NUM_ERB_BANDS)
            return
        }
        runCatching {
            val inputName = session.inputNames.firstOrNull() ?: return
            val fb = FloatBuffer.wrap(inFeatures)
            OnnxTensor.createTensor(env, fb, longArrayOf(1L, NUM_ERB_BANDS.toLong())).use { tensor ->
                session.run(mapOf(inputName to tensor)).use { result ->
                    val outTensor = result.firstOrNull()?.value as? OnnxTensor ?: return
                    val outBuf = outTensor.floatBuffer ?: return
                    for (b in 0 until NUM_ERB_BANDS) {
                        outMasks[b] = outBuf.get(b).coerceIn(0f, 1f)
                    }
                }
            }
        }.onFailure {
            System.arraycopy(inFeatures, 0, outMasks, 0, NUM_ERB_BANDS)
        }
    }
}
