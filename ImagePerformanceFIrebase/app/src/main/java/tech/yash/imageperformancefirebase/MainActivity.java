package tech.yash.imageperformancefirebase;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Point;
import android.graphics.Rect;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.firebase.ml.vision.FirebaseVision;
import com.google.firebase.ml.vision.common.FirebaseVisionImage;
import com.google.firebase.ml.vision.text.FirebaseVisionCloudTextRecognizerOptions;
import com.google.firebase.ml.vision.text.FirebaseVisionText;
import com.google.firebase.ml.vision.text.FirebaseVisionTextRecognizer;
import com.google.firebase.ml.vision.text.RecognizedLanguage;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private static final int RESULT_LOAD_IMG = 1;
    Button btnOpenGallery;
    ImageView ivImage;
    TextView tvConfidence;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        btnOpenGallery = (Button) findViewById(R.id.btnOpenGallery);
        ivImage = (ImageView) findViewById(R.id.ivImage);
        tvConfidence = (TextView) findViewById(R.id.tvConfidence);

        btnOpenGallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent photoPickerIntent = new Intent(Intent.ACTION_PICK);
                photoPickerIntent.setType("image/*");
                startActivityForResult(photoPickerIntent, RESULT_LOAD_IMG);
            }
        });
    }

    @Override
    protected void onActivityResult(int reqCode, int resultCode, Intent data) {
        super.onActivityResult(reqCode, resultCode, data);


        if (resultCode == RESULT_OK) {
            try {
                final Uri imageUri = data.getData();
                Log.d("OUTPUT", imageUri.toString());
                final InputStream imageStream = getContentResolver().openInputStream(imageUri);
                final Bitmap selectedImage = BitmapFactory.decodeStream(imageStream);
                ivImage.setImageBitmap(selectedImage);
                analyseImage(selectedImage);
            } catch (FileNotFoundException e) {
                e.printStackTrace();
                Toast.makeText(this, "Something went wrong", Toast.LENGTH_LONG).show();
            }

        }else {
            Toast.makeText(this, "You haven't picked Image",Toast.LENGTH_LONG).show();
        }
    }

    private void analyseImage(Bitmap bitmap){
        FirebaseVisionImage image;
        image = FirebaseVisionImage.fromBitmap(bitmap);

        FirebaseVisionCloudTextRecognizerOptions options = new FirebaseVisionCloudTextRecognizerOptions.Builder()
                .setModelType(FirebaseVisionCloudTextRecognizerOptions.DENSE_MODEL)
                .setLanguageHints(Arrays.asList("en"))
                .build();

        FirebaseVisionTextRecognizer detector = FirebaseVision.getInstance()
                .getCloudTextRecognizer(options);

        Task<FirebaseVisionText> result =
                detector.processImage(image)
                        .addOnSuccessListener(new OnSuccessListener<FirebaseVisionText>() {
                            @Override
                            public void onSuccess(FirebaseVisionText firebaseVisionText) {
                                String c = getConfidenceScore(firebaseVisionText);
                                Log.d("OUTPUT", c);
                                tvConfidence.setText(c);
                            }
                        })
                        .addOnFailureListener(
                                new OnFailureListener() {
                                    @Override
                                    public void onFailure(@NonNull Exception e) {
                                        e.printStackTrace();
                                        Toast.makeText(MainActivity.this, ""+e.getLocalizedMessage(), Toast.LENGTH_SHORT).show();
                                    }
                                });



    }

    private String getConfidenceScore(FirebaseVisionText result){

        float blockC = 0.0f;
        float lineC = 0.0f;
        float wordC = 0.0f;

        int b =0;
        int l =0;
        int w = 0;

        String output = "";

        String resultText = result.getText();
        for (FirebaseVisionText.TextBlock block: result.getTextBlocks()) {
            String blockText = block.getText();
            Float blockConfidence = block.getConfidence();
            List<RecognizedLanguage> blockLanguages = block.getRecognizedLanguages();
            Point[] blockCornerPoints = block.getCornerPoints();
            Rect blockFrame = block.getBoundingBox();
            blockC += blockConfidence;
            b++;
            for (FirebaseVisionText.Line line: block.getLines()) {
                String lineText = line.getText();
                Float lineConfidence = line.getConfidence();
                List<RecognizedLanguage> lineLanguages = line.getRecognizedLanguages();
                Point[] lineCornerPoints = line.getCornerPoints();
                Rect lineFrame = line.getBoundingBox();
                lineC+= lineConfidence;
                l++;
                for (FirebaseVisionText.Element element: line.getElements()) {
                    String elementText = element.getText();
                    Float elementConfidence = element.getConfidence();
                    List<RecognizedLanguage> elementLanguages = element.getRecognizedLanguages();
                    Point[] elementCornerPoints = element.getCornerPoints();
                    Rect elementFrame = element.getBoundingBox();
                    wordC+= elementConfidence;
                    w++;
                }
            }
        }

        output = "Confidence \n Block:"+blockC/b+", "+b+"\n "+
                "Line:"+lineC/l+", "+l+"\n "+
                "Word:"+wordC/w+", "+w;

        return output;

    }

}
