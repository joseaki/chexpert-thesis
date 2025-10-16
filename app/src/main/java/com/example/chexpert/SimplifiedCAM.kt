package com.example.chexpert
// Updated SimplifiedCAM that loads NumPy weights
// This is the most reliable method!

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.Log
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

class SimplifiedCAM(private val context: Context) {

    companion object {
        private const val TAG = "SimplifiedCAM"
    }

    private var module: Module? = null
    private var classifierWeights: FloatArray? = null
    private var numClasses = 1
    private var numFeatures = 1024

    fun loadModel(): Boolean {
        return try {
            Log.d(TAG, "=== Loading Model ===")

            // Load PyTorch model
            val modelPath = try {
                assetFilePath(context, "model_mobile.ptl")
            } catch (e: Exception) {
                Log.d(TAG, "Trying .pt format...")
                assetFilePath(context, "model_mobile.pt")
            }

            module = Module.load(modelPath)
            Log.d(TAG, "✓ Model loaded: $modelPath")

            // Load weights from simple binary file
            classifierWeights = loadBinaryWeights()

            if (classifierWeights != null) {
                Log.d(TAG, "✓ Weights loaded: [${numClasses}, ${numFeatures}]")
                Log.d(TAG, "✓ Total weight values: ${classifierWeights!!.size}")
                Log.d(TAG, "=== Ready! ===")
                true
            } else {
                Log.e(TAG, "❌ Failed to load weights")
                false
            }

        } catch (e: Exception) {
            Log.e(TAG, "❌ Error loading model: ${e.message}", e)
            e.printStackTrace()
            false
        }
    }

    private fun loadBinaryWeights(): FloatArray? {
        try {
            Log.d(TAG, "Loading binary weights...")

            // Read the binary file
            val inputStream = context.assets.open("weights.bin")
            val allBytes = inputStream.readBytes()
            inputStream.close()

            Log.d(TAG, "Read ${allBytes.size} bytes from weights.bin")

            // Wrap in ByteBuffer with LITTLE_ENDIAN order
            val buffer = ByteBuffer.wrap(allBytes).order(ByteOrder.LITTLE_ENDIAN)

            // Read header: 2 integers (num_classes, num_features)
            numClasses = buffer.int
            numFeatures = buffer.int

            Log.d(TAG, "Header: classes=$numClasses, features=$numFeatures")

            // Calculate expected size
            val expectedFloats = numClasses * numFeatures
            val expectedBytes = 8 + (expectedFloats * 4)  // 8 bytes header + floats

            if (allBytes.size != expectedBytes) {
                Log.w(TAG, "Warning: File size mismatch. Expected $expectedBytes, got ${allBytes.size}")
            }

            // Read all weight values
            val weights = FloatArray(expectedFloats)
            for (i in 0 until expectedFloats) {
                if (buffer.remaining() >= 4) {
                    weights[i] = buffer.float
                } else {
                    Log.e(TAG, "Ran out of data at index $i")
                    return null
                }
            }

            Log.d(TAG, "✓ Loaded ${weights.size} weight values")
            Log.d(TAG, "  First 5 weights: ${weights.take(5)}")

            return weights

        } catch (e: Exception) {
            Log.e(TAG, "Failed to load binary weights: ${e.message}", e)
            e.printStackTrace()
            return null
        }
    }

    fun generateCAM(bitmap: Bitmap, classIdx: Int = 0): Result {
        if (module == null || classifierWeights == null) {
            throw IllegalStateException("Model not loaded. Call loadModel() first.")
        }

        Log.d(TAG, "Generating CAM for class $classIdx...")
        val startTime = System.currentTimeMillis()

        // Resize to 224x224
        val resized = Bitmap.createScaledBitmap(bitmap, 224, 224, true)

        // Convert to tensor
        val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
            resized,
            floatArrayOf(0f, 0f, 0f),  // mean
            floatArrayOf(1f, 1f, 1f)   // std
        )

        // Run inference
        val output = module!!.forward(IValue.from(inputTensor))

        if (!output.isTuple) {
            throw Exception("Model should return (predictions, features) tuple")
        }

        val tuple = output.toTuple()
        val predictions = tuple[0].toTensor()
        val features = tuple[1].toTensor()

        val scores = predictions.dataAsFloatArray
        val score = scores[classIdx]

        Log.d(TAG, "Prediction score: $score")

        // Compute CAM
        val cam = computeCAM(features, classIdx)

        // Create overlay
        val overlayBitmap = createCAMOverlay(resized, cam)

        val processingTime = System.currentTimeMillis() - startTime
        Log.d(TAG, "✓ CAM generated in ${processingTime}ms")

        return Result(overlayBitmap, score, processingTime)
    }

    private fun computeCAM(featureTensor: Tensor, classIdx: Int): Array<FloatArray> {
        val shape = featureTensor.shape()
        val batchSize = shape[0].toInt()
        val channels = shape[1].toInt()
        val height = shape[2].toInt()
        val width = shape[3].toInt()

        Log.d(TAG, "Feature tensor shape: [$batchSize, $channels, $height, $width]")

        val features = featureTensor.dataAsFloatArray
        val cam = Array(height) { FloatArray(width) { 0f } }

        // Get weights for target class
        val weightStartIdx = classIdx * numFeatures

        // Weighted sum of feature maps
        for (c in 0 until channels) {
            val weight = classifierWeights!![weightStartIdx + c]

            for (h in 0 until height) {
                for (w in 0 until width) {
                    val featureIdx = c * height * width + h * width + w
                    cam[h][w] += weight * features[featureIdx]
                }
            }
        }

        // Apply ReLU (remove negative activations)
        for (h in 0 until height) {
            for (w in 0 until width) {
                cam[h][w] = kotlin.math.max(0f, cam[h][w])
            }
        }

        // Normalize to [0, 1]
        var minVal = Float.MAX_VALUE
        var maxVal = Float.MIN_VALUE

        for (row in cam) {
            for (value in row) {
                minVal = kotlin.math.min(minVal, value)
                maxVal = kotlin.math.max(maxVal, value)
            }
        }

        val range = maxVal - minVal
        if (range > 1e-8f) {
            for (h in cam.indices) {
                for (w in cam[0].indices) {
                    cam[h][w] = (cam[h][w] - minVal) / range
                }
            }
        }

        Log.d(TAG, "CAM range: [$minVal, $maxVal]")

        return cam
    }

    private fun createCAMOverlay(original: Bitmap, cam: Array<FloatArray>): Bitmap {
        val camHeight = cam.size
        val camWidth = cam[0].size

        // Create heatmap
        val heatmap = Bitmap.createBitmap(camWidth, camHeight, Bitmap.Config.ARGB_8888)

        for (h in 0 until camHeight) {
            for (w in 0 until camWidth) {
                val intensity = cam[h][w]
                val color = jetColormap(intensity)
                heatmap.setPixel(w, h, color)
            }
        }

        // Resize heatmap to match original
        val resizedHeatmap = Bitmap.createScaledBitmap(
            heatmap,
            original.width,
            original.height,
            true
        )

        // Create result bitmap
        val result = Bitmap.createBitmap(
            original.width,
            original.height,
            Bitmap.Config.ARGB_8888
        )

        val canvas = Canvas(result)

        // Draw original image
        canvas.drawBitmap(original, 0f, 0f, null)

        // Overlay heatmap with transparency
        val paint = Paint().apply {
            alpha = 128  // 50% transparent
        }
        canvas.drawBitmap(resizedHeatmap, 0f, 0f, paint)

        return result
    }

    private fun jetColormap(value: Float): Int {
        // Jet colormap: blue -> cyan -> green -> yellow -> red
        val v = value.coerceIn(0f, 1f)

        val r: Int
        val g: Int
        val b: Int

        when {
            v < 0.25f -> {
                r = 0
                g = (255 * (v / 0.25f)).toInt()
                b = 255
            }
            v < 0.5f -> {
                r = 0
                g = 255
                b = (255 * (1 - (v - 0.25f) / 0.25f)).toInt()
            }
            v < 0.75f -> {
                r = (255 * ((v - 0.5f) / 0.25f)).toInt()
                g = 255
                b = 0
            }
            else -> {
                r = 255
                g = (255 * (1 - (v - 0.75f) / 0.25f)).toInt()
                b = 0
            }
        }

        return Color.argb(255, r, g, b)
    }

    private fun assetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)

        // If already copied, return path
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }

        // Copy from assets to internal storage
        context.assets.open(assetName).use { inputStream ->
            file.outputStream().use { outputStream ->
                inputStream.copyTo(outputStream)
            }
        }

        Log.d(TAG, "Copied $assetName to ${file.absolutePath}")
        return file.absolutePath
    }

    data class Result(
        val bitmap: Bitmap,
        val score: Float,
        val processingTimeMs: Long
    )
}