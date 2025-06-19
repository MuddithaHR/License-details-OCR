import json
import os
import time
from pipeline import detail_extraction_pipeline, load_models
    
if __name__ == "__main__":

    img_file_path = "C:/Users/ASUS/Downloads/New folder (3)/qvv1.jpg"# your image path here

    #Load models  
    yolo_model, ocr_model = load_models()

    feedback_message = detail_extraction_pipeline(yolo_model, ocr_model, img_file_path)

    # if feed message is 'Detection Successful.' or 'Some rows are missing in the result.', then detected data is stored in 'outputs/' folder. Else only feedback message.
    print(feedback_message)



#------------------------------------------------------
#------------------------------------------------------
################# FOR GIVEN IMAGES ####################
############ CAN USE FOR MULTIPLE IMAGES ##############
#------------------------------------------------------
#------------------------------------------------------


#     # Load models  
#     yolo_model, ocr_model = load_models()
#     folder_path = 'C:/Users/ASUS/Downloads/Images'

#     processed_images_dict = {}

#     for file in sorted(os.listdir(folder_path)):
#         if file in a:
#             img_file_path = folder_path + '/' + file
#             feedback_message = detail_extraction_pipeline(yolo_model, ocr_model, img_file_path)
#             print(feedback_message)

#         if feedback_message in processed_images_dict:
#             processed_images_dict[feedback_message][0] += 1
#             processed_images_dict[feedback_message][1].append(file)
#         else:
#             processed_images_dict[feedback_message] = [1, [file]]

# with open("outputs/Sample Data/results_summary/output.json", "w") as f:
#     json.dump(processed_images_dict, f, indent=4)

#     print(time.time() - t)

