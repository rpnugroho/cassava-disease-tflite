# Cassava Disease Indentification on Android
This is simple implementation of image classification on android using TF Lite. We train our model using tflite-model-maker and create android app.\
![Demo](/docs/demo.gif)

## Data
You can find this dataset in [Cassava Leaf Disease Classification](https://www.kaggle.com/c/cassava-leaf-disease-classification/overview).\
The dataset consists of leaf images of the cassava plant, 21,367 labeled images collected during a regular survey in Uganda. Most images were crowdsourced from farmers taking photos of their gardens, and annotated by experts at the National Crops Resources Research Institute (NaCRRI) in collaboration with the AI lab at Makerere University, Kampala.\
There are 5 labels on this dataset: "Cassava Bacterial Blight (CBB)", "Cassava Brown Streak Disease (CBSD)", "Cassava Green Mottle (CGM)", "Cassava Mosaic Disease (CMD)", "Healthy".

## Model
We use [`tflite-model-maker`](https://pypi.org/project/tflite-model-maker/) to create efficientnet_lite model.\
We train efficientnet_lite 0-4 for 5 epochs and found that efficientnet_lite3 seems to be the best choice.\
![Image of loss](/docs/loss_and_val_loss.png)\
![Image of acc](/docs/acc_and_val_acc.png)\
After that we train efficientnet_lite3 with this parameters for 15 epochs,
``` python
(
	...
	learning_rate=5e-4,
	warmup_steps=2*534,
	train_whole_model=True,
	use_augmentation=True,
	dropout_rate=0,
	shuffle=True,
	...
)
```
as a result, model accuracy increased (epoch 11-15).\
![Image of fine tune](/docs/fine_tune_acc_loss.png)\
We also test our model with test_data (hold out data) and got **88%** accuracy.
<br>
> Google has released [CropNet: Cassava Disease Detection](https://tfhub.dev/google/cropnet/classifier/cassava_disease_V1/2), you can use this model rather than creating on your own.

## Android App
After training done, we export this model and got two output files: model.tflite and labels.txt.\
We follow this [codelab](https://codelabs.developers.google.com/codelabs/recognize-flowers-with-tensorflow-on-android), copy our model to `.../android/app/src/main/assets/` and build using android studio.\
If you get an error add this script to your `build.gradle` file.
``` java
dependencies {
	
    ...
    // Build off of nightly TensorFlow Lite
    // TODO: Add TFLite dependencies
    implementation('org.tensorflow:tensorflow-lite:0.0.0-nightly') { changing = true }
    implementation('org.tensorflow:tensorflow-lite-gpu:0.0.0-nightly') { changing = true }
    implementation('org.tensorflow:tensorflow-lite-support:0.0.0-nightly') { changing = true }
    // Use local TensorFlow library
    // implementation 'org.tensorflow:tensorflow-lite-local:0.0.0'
    androidTestImplementation 'androidx.test.ext:junit:1.1.0'
    androidTestImplementation 'com.android.support.test:rules:1.1.0'
    androidTestImplementation 'com.google.truth:truth:0.43'
    testImplementation 'com.google.truth:truth:0.43'
}
```
## References
- https://www.kaggle.com/c/cassava-leaf-disease-classification/overview
- https://www.tensorflow.org/lite/tutorials/model_maker_image_classification
- https://codelabs.developers.google.com/codelabs/recognize-flowers-with-tensorflow-on-android


