package com.example.chexpert

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Button
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.example.chexpert.ui.theme.ChexpertTheme
import androidx.core.graphics.createBitmap
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.InputStream

class MainActivity : ComponentActivity() {
    private lateinit var cam: SimplifiedCAM

    @OptIn(ExperimentalLayoutApi::class)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        // Initialize loader
        val atelectasisLoader = PyTorchModelLoader()
        val cardiomegalyLoader = PyTorchModelLoader(1080)
        val consolidationLoader = PyTorchModelLoader()
        val edemaLoader = PyTorchModelLoader(1024)
        val effusionLoader = PyTorchModelLoader(448)

        cam = SimplifiedCAM(this)
        // Load model in background
//        loadModelAsync()
        cam.loadModel()


        atelectasisLoader.loadModelFromAssets(this, "atelectasis_final_model.ptl")
        cardiomegalyLoader.loadModelFromAssets(this, "cardiomegaly_final_model_densenet_201_v3_1080.ptl")
        consolidationLoader.loadModelFromAssets(this, "consolidation_final_model.ptl")
        edemaLoader.loadModelFromAssets(this, "edema_final_model_densenet_201_v3_1024.ptl")
        effusionLoader.loadModelFromAssets(this, "effusion_final_model_densenet_201_v4_448.ptl")

        // Load image from file
//        val bitmap = BitmapFactory.decodeFile("")


        // Run prediction

//
//        // Get top prediction
//        val maxIndex = results.indices.maxByOrNull { results[it] } ?: 0
//        println("Predicted class: $maxIndex with confidence: ${results[maxIndex]}")
//        var bitmap: Bitmap? = createBitmap(100, 100)
        setContent {
            var bitmap by remember { mutableStateOf<android.graphics.Bitmap?>(null) }
            var predictions by remember { mutableStateOf<FloatArray>(floatArrayOf()) }
            val labels = listOf("Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion")

            ChexpertTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    Column {
                        Greeting(
                            "Android", Modifier.padding(innerPadding)
                        )
                        FlowRow (
                            modifier = Modifier.padding(16.dp),
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            verticalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            ButtonInference("CAM") {
                                var image = loadImage("00000001_000.png", this@MainActivity)
                                bitmap = image;
                                val (heatmap, score) = cam.generateCAM(image)
                                bitmap = heatmap;
                                predictions = floatArrayOf(score)

                            }
                            ButtonInference("Ejemplo 1", {
                                var image = loadImage("00000001_000.png", this@MainActivity)
                                bitmap = image;
                                predictions = runInference(
                                    image,
                                    atelectasisLoader = atelectasisLoader,
                                    cardiomegalyLoader = cardiomegalyLoader,
                                    consolidationLoader = consolidationLoader,
                                    edemaLoader = edemaLoader,
                                    effusionLoader = effusionLoader
                                ) ?: floatArrayOf()
                            })
                            ButtonInference("Ejemplo 2", {
                                val image = loadImage("00000012_000.png", this@MainActivity)
                                bitmap = image;
                                predictions = runInference(
                                    image,
                                    atelectasisLoader = atelectasisLoader,
                                    cardiomegalyLoader = cardiomegalyLoader,
                                    consolidationLoader = consolidationLoader,
                                    edemaLoader = edemaLoader,
                                    effusionLoader = effusionLoader
                                ) ?: floatArrayOf()
                            })
                            ButtonInference("Ejemplo 3", {
                                val image = loadImage("00000005_006.png", this@MainActivity)
                                bitmap = image;
                                predictions = runInference(
                                    image,
                                    atelectasisLoader = atelectasisLoader,
                                    cardiomegalyLoader = cardiomegalyLoader,
                                    consolidationLoader = consolidationLoader,
                                    edemaLoader = edemaLoader,
                                    effusionLoader = effusionLoader
                                ) ?: floatArrayOf()
                            })
                            ButtonInference("Ejemplo 4", {
                                val image = loadImage("00000013_027.png", this@MainActivity)
                                bitmap = image;
                                predictions = runInference(
                                    image,
                                    atelectasisLoader = atelectasisLoader,
                                    cardiomegalyLoader = cardiomegalyLoader,
                                    consolidationLoader = consolidationLoader,
                                    edemaLoader = edemaLoader,
                                    effusionLoader = effusionLoader
                                ) ?: floatArrayOf()
                            })
                            ButtonInference("Ejemplo 5", {
                                val image = loadImage("00000013_045.png", this@MainActivity)
                                bitmap = image;
                                predictions = runInference(
                                    image,
                                        atelectasisLoader = atelectasisLoader,
                                        cardiomegalyLoader = cardiomegalyLoader,
                                        consolidationLoader = consolidationLoader,
                                        edemaLoader = edemaLoader,
                                        effusionLoader = effusionLoader
                                    ) ?: floatArrayOf()
                                })
                        }
                        bitmap?.let {
                            Image(
                                bitmap = it.asImageBitmap(),
                                contentDescription = "Loaded image",
                                modifier = Modifier
                                    .size(375.dp)
                            )
                        }
                        predictions.forEachIndexed { index, number ->
                            Text(text = "${labels[index]}: $number")
                        }
                    }
                }
            }
        }
    }

    private fun loadModelAsync() {
        lifecycleScope.launch(Dispatchers.IO) {
            val success = cam.loadModel()
            println("Success")

            withContext(Dispatchers.Main) {
                if (success) {
                    Toast.makeText(
                        this@MainActivity,
                        "Model loaded successfully!",
                        Toast.LENGTH_SHORT
                    ).show()
                } else {
                    Toast.makeText(
                        this@MainActivity,
                        "Failed to load model",
                        Toast.LENGTH_LONG
                    ).show()
                }
            }
        }
    }
}

fun loadImage(imageName: String, context: Context,): Bitmap {
    val inputStream: InputStream = context.assets.open(imageName)
    val bitmap = BitmapFactory.decodeStream(inputStream)
    inputStream.close()
    return bitmap
}

fun runInference(
    bitmap: Bitmap,
    atelectasisLoader: PyTorchModelLoader,
    cardiomegalyLoader: PyTorchModelLoader,
    consolidationLoader: PyTorchModelLoader,
    edemaLoader: PyTorchModelLoader,
    effusionLoader: PyTorchModelLoader
): FloatArray? {
    val atelectasis = atelectasisLoader.predict(bitmap)
    val cardiomegaly = cardiomegalyLoader.predict(bitmap)
    val consolidation = consolidationLoader.predict(bitmap)
    val edema = edemaLoader.predict(bitmap)
    val effusion = effusionLoader.predict(bitmap)

    val arrays = listOf(
        atelectasis,
        cardiomegaly,
        consolidation,
        edema,
        effusion
    )

    val predictions = arrays.flatMap { it.toList() }.toFloatArray()

    println("atelectasis: ${atelectasis.contentToString()}")
    println("cardiomegaly: ${cardiomegaly.contentToString()}")
    println("consolidation: ${consolidation.contentToString()}")
    println("edema: ${edema.contentToString()}")
    println("effusion: ${effusion.contentToString()}")

    return predictions
}

@Composable
fun Greeting(name: String, modifier: Modifier = Modifier) {
    Text(
        text = "Hello $name!",
        modifier = modifier
    )
}

@Composable
fun ButtonInference(text:String, onClick: () -> Unit) {
    Button(onClick = { onClick() }) {
        Text(text)
    }
}

@Preview(showBackground = true)
@Composable
fun GreetingPreview() {
    ChexpertTheme {
        Greeting("Android")
    }
}