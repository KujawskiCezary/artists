import tensorflow as tf
from main import cnn_model_fn, load_image

def test(unused_argv):
  image = {'author' : 'JanMatejko',
           'name' : 'Sermon_of_Skarga.jpg'}
  eval_data, eval_labels = load_image(image)
  
  # Create the Estimator
  classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="tmp/convnet_model")
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)
   
    
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)

    


if __name__ == "__main__":
  tf.app.run(test)