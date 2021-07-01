package com.example.imagepro;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

import androidx.fragment.app.Fragment;

import org.checkerframework.checker.units.qual.A;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.examples.posenet.lib.Device;
import org.tensorflow.lite.examples.posenet.lib.Person;
import org.tensorflow.lite.examples.posenet.lib.Posenet;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;



public class objectDetectorClass extends Fragment {
    private SurfaceHolder surfaceHolder = null;

    /** Paint class holds the style and color information to draw geometries,text and bitmaps. */
    private Paint paint =new  Paint();

 int MODEL_WIDTH = 257;
int  MODEL_HEIGHT = 257;
    // this is used to load model and predict
    private Interpreter interpreter;
    // store all label in array
    private List<String> labelList;
    private int INPUT_SIZE;
    private int PIXEL_SIZE=3; // for RGB
    private int IMAGE_MEAN=0;
    private  float IMAGE_STD=255.0f;
    // use to initialize gpu in app
    private GpuDelegate gpuDelegate;
    private int height=0;
    private  int width=0;
    private Context context;

    objectDetectorClass(AssetManager assetManager, String modelPath, String labelPath, int inputSize, Context activityContext) throws IOException{
        INPUT_SIZE=inputSize;
        // use to define gpu or cpu // no. of threads
        Interpreter.Options options=new Interpreter.Options();
        gpuDelegate=new GpuDelegate();
        options.addDelegate(gpuDelegate);
        options.setNumThreads(4); // set it according to your phone
        // loading model
        interpreter=new Interpreter(loadModelFile(assetManager,modelPath),options);
        // load labelmap
        labelList=loadLabelList(assetManager,labelPath);
        context=activityContext;
    }

    private List<String> loadLabelList(AssetManager assetManager, String labelPath) throws IOException {
        // to store label
        List<String> labelList=new ArrayList<>();
        // create a new reader
        BufferedReader reader=new BufferedReader(new InputStreamReader(assetManager.open(labelPath)));
        String line;
        // loop through each line and store it to labelList
        while ((line=reader.readLine())!=null){
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }

    private ByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        // use to get description of file
        AssetFileDescriptor fileDescriptor=assetManager.openFd(modelPath);
        FileInputStream inputStream=new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel=inputStream.getChannel();
        long startOffset =fileDescriptor.getStartOffset();
        long declaredLength=fileDescriptor.getDeclaredLength();

        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declaredLength);
    }
    // create new Mat function
    public Mat recognizeImage(Mat mat_image){
        // Rotate original image by 90 degree get get portrait frame
        Mat rotated_mat_image=new Mat();
        Core.flip(mat_image.t(),rotated_mat_image,1);
        // if you do not do this process you will get improper prediction, less no. of object
        // now convert it to bitmap
        Bitmap bitmap=null;
        bitmap=Bitmap.createBitmap(rotated_mat_image.cols(),rotated_mat_image.rows(),Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(rotated_mat_image,bitmap);
        // define height and width
        height=bitmap.getHeight();
        width=bitmap.getWidth();

        // scale the bitmap to input size of model
         Bitmap scaledBitmap=Bitmap.createScaledBitmap(bitmap,INPUT_SIZE,INPUT_SIZE,false);

         // convert bitmap to bytebuffer as model input should be in it
        ByteBuffer byteBuffer=convertBitmapToByteBuffer(scaledBitmap);

        // defining output
        // 10: top 10 object detected
        // 4: there coordinate in image
      //  float[][][]result=new float[1][10][4];
        Object[] input=new Object[1];
        input[0]=byteBuffer;

        Map<Integer,Object> output_map=new TreeMap<>();
        // we are not going to use this method of output
        // instead we create treemap of three array (boxes,score,classes)

        float[][][]boxes =new float[1][10][4];
        // 10: top 10 object detected
        // 4: there coordinate in image
        float[][] scores=new float[1][10];
        // stores scores of 10 object
        float[][] classes=new float[1][10];
        // stores class of object

        // add it to object_map;
        output_map.put(0,boxes);
        output_map.put(1,classes);
        output_map.put(2,scores);

        // now predict
        interpreter.runForMultipleInputsOutputs(input,output_map);


        Object value=output_map.get(0);
        Object Object_class=output_map.get(1);
        Object score=output_map.get(2);

        // loop through each object
        // as output has only 10 boxes
        for (int i=0;i<10;i++){
            float class_value=(float) Array.get(Array.get(Object_class,0),i);
            float score_value=(float) Array.get(Array.get(score,0),i);
            // define threshold for score
            if(score_value>0.5){
                Object box1=Array.get(Array.get(value,0),i);
                // we are multiplying it with Original height and width of frame

                float top=(float) Array.get(box1,0)*height;
                float left=(float) Array.get(box1,1)*width;
                float bottom=(float) Array.get(box1,2)*height;
                float right=(float) Array.get(box1,3)*width;
//                 draw rectangle in Original frame //  starting point    // ending point of box  // color of box       thickness

                // write text on frame
                                                // string of class name of object  // starting point                         // color of text           // size of text

                if((labelList.get((int) class_value)).equals("person")) {
                    Imgproc.putText(rotated_mat_image,labelList.get((int) class_value),new Point(left,top),3,1,new Scalar(255, 0, 0, 255),2);
                    Imgproc.rectangle(rotated_mat_image,new Point(left,top),new Point(right,bottom),new Scalar(0, 255, 0, 255),2);
                    Log.v("MM","Matched");
                    processImage(scaledBitmap);

                }
                else{
                    Log.v("MM","Not Matched"+labelList.get((int) class_value));
                }

            }

        }
        // select device and run

        // before returning rotate back by -90 degree
        Core.flip(rotated_mat_image.t(),mat_image,0);


        return mat_image;
    }


    /** Process image using Posenet library.   */
      public void processImage(Bitmap bitmap ) {
        // Crop bitmap.
          Posenet pose=new Posenet(context,"posenet_model.tflite",Device.GPU);
Log.e("PROCESs","PROCESs");

        Bitmap croppedBitmap = pose.cropBitmap(bitmap);
          Log.e("CROP","CROP");

        // Created scaled version of bitmap for model input.
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(croppedBitmap, MODEL_WIDTH, MODEL_HEIGHT, true);
          Log.e("SCALE","SCALE");

        // Perform inference.
          Person person=pose.estimateSinglePose(scaledBitmap);
          Log.e("AFTER SCALE","AFTER SCALE");
//        Canvas canvas = surfaceHolder.lockCanvas();
//        draw(canvas, person, scaledBitmap);


      }

    /** Draw bitmap on Canvas.   */
//    public draw( Canvas canvas , Person person , Bitmap bitmap) {
//
//        Log.v("TAG", "index=" + (person.keyPoints[0].position.x));
//
//
////
////
////        canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)
////        // Draw `bitmap` and `person` in square canvas.
////        int screenWidth;
////        int screenHeight;
////        int left;
////        int right;
////        int top;
////        int bottom;
////        if (canvas.height > canvas.width) {
////            screenWidth = canvas.width;
////            screenHeight = canvas.width;
////            left = 0;
////            top = (canvas.height - canvas.width) / 2;
////        } else {
////            screenWidth = canvas.height;
////            screenHeight = canvas.height;
////            left = (canvas.width - canvas.height) / 2;
////            top = 0;
////        }
////        right = left + screenWidth;
////        bottom = top + screenHeight;
////
////        setPaint();
////        canvas.drawBitmap(
////                bitmap,
////                Rect(0, 0, bitmap.width, bitmap.height),
////                Rect(left, top, right, bottom),
////                paint
////        )
////
////        val widthRatio = screenWidth.toFloat() / MODEL_WIDTH
////        val heightRatio = screenHeight.toFloat() / MODEL_HEIGHT
////
////        // Draw key points over the image.
////        for (keyPoint in person.keyPoints) {
////            if (keyPoint.score > minConfidence) {
////                val position = keyPoint.position
////                val adjustedX: Float = position.x.toFloat() * widthRatio + left
////                val adjustedY: Float = position.y.toFloat() * heightRatio + top
////                canvas.drawCircle(adjustedX, adjustedY, circleRadius, paint)
////            }
////        }
////
////        for (line in bodyJoints) {
////            if (
////                    (person.keyPoints[line.first.ordinal].score > minConfidence) and
////                    (person.keyPoints[line.second.ordinal].score > minConfidence)
////      ) {
////                canvas.drawLine(
////                        person.keyPoints[line.first.ordinal].position.x.toFloat() * widthRatio + left,
////                        person.keyPoints[line.first.ordinal].position.y.toFloat() * heightRatio + top,
////                        person.keyPoints[line.second.ordinal].position.x.toFloat() * widthRatio + left,
////                        person.keyPoints[line.second.ordinal].position.y.toFloat() * heightRatio + top,
////                        paint
////                )
////            }
////        }
////
////        surfaceHolder!!.unlockCanvasAndPost(canvas);
//    }


//    /** Set the paint color and size.    */
//    private  void setPaint() {
//        paint.color = Color.RED;
//        paint.textSize = 80.0f;
//        paint.strokeWidth = 8.0f;
//    }
    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer;
        // some model input should be quant=0  for some quant=1
        // for this quant=0

        int quant=0;
        int size_images=INPUT_SIZE;
        if(quant==0){
            byteBuffer=ByteBuffer.allocateDirect(1*size_images*size_images*3);
        }
        else {
            byteBuffer=ByteBuffer.allocateDirect(4*1*size_images*size_images*3);
        }
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues=new int[size_images*size_images];
        bitmap.getPixels(intValues,0,bitmap.getWidth(),0,0,bitmap.getWidth(),bitmap.getHeight());
        int pixel=0;

        // some error
        //now run
        for (int i=0;i<size_images;++i){
            for (int j=0;j<size_images;++j){
                final  int val=intValues[pixel++];
                if(quant==0){
                    byteBuffer.put((byte) ((val>>16)&0xFF));
                    byteBuffer.put((byte) ((val>>8)&0xFF));
                    byteBuffer.put((byte) (val&0xFF));
                }
                else {
                    // paste this
                    byteBuffer.putFloat((((val >> 16) & 0xFF))/255.0f);
                    byteBuffer.putFloat((((val >> 8) & 0xFF))/255.0f);
                    byteBuffer.putFloat((((val) & 0xFF))/255.0f);
                }
            }
        }
    return byteBuffer;
    }
}
