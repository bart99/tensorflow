package com.example.shpark.mnist;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.TypedValue;
import android.view.Display;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.nio.ByteBuffer;

import static android.R.id.edit;

public class MainActivity extends AppCompatActivity implements View.OnTouchListener {

    static {
        System.loadLibrary("tensorflow_inference");
    }

    private static final String MODEL_FILE = "file:///android_asset/mnist.pb";
    private static final String INPUT_NODE = "input";
    private static final String OUTPUT_NODE = "predict_op";
    private static final int[] INPUT_SIZE = {28, 28};

    private TensorFlowInferenceInterface inferenceInterface = null;
    private ImageView imageView = null;
    private ImageView resizedImageView = null;
    private EditText editValue = null;
    private Bitmap bitmap;
    private Bitmap resizeBitmap;
    private Canvas canvas;
    private Paint paint;
    private float startX = 0, startY = 0, endX = 0, endY = 0;
    private Path path;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        initMnistModel();

        imageView = (ImageView) this.findViewById(R.id.imageView);
        resizedImageView = (ImageView) this.findViewById(R.id.resizedImageView);

        editValue = (EditText)this.findViewById(R.id.editText);

        int px = (int)TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, 350, getResources().getDisplayMetrics());
        bitmap = Bitmap.createBitmap(px, px, Bitmap.Config.ARGB_8888);
        bitmap.eraseColor(Color.WHITE);
        canvas = new Canvas(bitmap);
        paint = new Paint();
        paint.setColor(Color.BLACK);
        paint.setStrokeWidth(35.f);
        paint.setAntiAlias(true);
        paint.setDither(true);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeJoin(Paint.Join.ROUND);
        paint.setStrokeCap(Paint.Cap.ROUND);
        imageView.setImageBitmap(bitmap);

        resizeBitmap = Bitmap.createScaledBitmap(bitmap, 28, 28, true);
        resizedImageView.setImageBitmap(resizeBitmap);

        imageView.setOnTouchListener(this);

        Button predictBtn = (Button)this.findViewById(R.id.predictBtn);
        predictBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                Bitmap trimBitmap = trim(bitmap, Color.WHITE);

                resizeBitmap = Bitmap.createScaledBitmap(trimBitmap, 28, 28, true);

                int x = resizeBitmap.getWidth();
                int y = resizeBitmap.getHeight();
                int[] intArray = new int[x * y];
                resizeBitmap.getPixels(intArray, 0, x, 0, 0, x, y);

                float[] floatArray = new float[intArray.length];
                for(int i=0; i<intArray.length; i++) {
                    floatArray[i] = intArray[i];
                }

                float[] keep_conv = {1.0f};
                float[] keep_hidden = {1.0f};
                int[] res = {0};
                inferenceInterface.feed(INPUT_NODE, floatArray, 1, 28, 28, 1);
                inferenceInterface.feed("keep_conv", keep_conv);
                inferenceInterface.feed("keep_hidden", keep_hidden);
                inferenceInterface.run(new String[] {OUTPUT_NODE});
                inferenceInterface.fetch(OUTPUT_NODE, res);

                editValue.setText(String.valueOf(res[0]));

                resizedImageView.setImageBitmap(resizeBitmap);
                resizedImageView.invalidate();
            }
        });

        Button clearBtn = (Button)this.findViewById(R.id.clearBtn);
        clearBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                bitmap.eraseColor(Color.WHITE);
                editValue.setText("");
            }
        });
    }

    private void initMnistModel() {
        inferenceInterface = new TensorFlowInferenceInterface(getAssets(), MODEL_FILE);
    }

    @Override
    public boolean onTouch(View v, MotionEvent event) {
        int action = event.getAction();
        switch (action) {
            case MotionEvent.ACTION_DOWN:
                startX=event.getX();
                startY=event.getY();
                break;
            case MotionEvent.ACTION_MOVE:
                endX = event.getX();
                endY = event.getY();
                canvas.drawLine(startX,startY,endX,endY, paint);
                imageView.invalidate();
                startX=endX;
                startY=endY;
                break;
            case MotionEvent.ACTION_UP:
                break;
            case MotionEvent.ACTION_CANCEL:
                break;
            default:
                break;
        }
        return true;
    }

    private Bitmap trim(Bitmap bitmap, int trimColor){
        int minX = Integer.MAX_VALUE;
        int maxX = 0;
        int minY = Integer.MAX_VALUE;
        int maxY = 0;

        for(int x = 0; x < bitmap.getWidth(); x++){
            for(int y = 0; y < bitmap.getHeight(); y++){
                if(bitmap.getPixel(x, y) != trimColor){
                    if(x < minX){
                        minX = x;
                    }
                    if(x > maxX){
                        maxX = x;
                    }
                    if(y < minY){
                        minY = y;
                    }
                    if(y > maxY){
                        maxY = y;
                    }
                }
            }
        }

        int width = maxX - minX + 1;
        int height = maxY - minY + 1;

        int size = width > height ? width : height;
        int bitmapSize = size + 10;

        if (size == width) {
            minY = (bitmap.getHeight() - size) / 2;
        } else {
            minX = (bitmap.getWidth() - size) / 2;
        }

        return Bitmap.createBitmap(bitmap, minX, minY, size, size);
    }
}
