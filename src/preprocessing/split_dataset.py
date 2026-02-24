import os
import random
import shutil


SOURCE_DATA_PATH = "/mnt/A4F0E4F6F0E4D01A/Shams Iqbal/VS code/Kaggle/Datasets/CIFAR-100-dataset/train (Copy)"
DESTINATION_DATAPATH = "/mnt/A4F0E4F6F0E4D01A/Shams Iqbal/VS code/Kaggle/Datasets/CIFAR-100-dataset/val"

all_files = []

class ValidationData:
    def __init__(self):
        pass
    
    def _does_exist(self, path=DESTINATION_DATAPATH):
        try:
            if not os.path.exists(path):
                os.mkdir(path)
        except:
            raise Exception(f"{path} folder could not be created")
        

    def _choose_file_index(self, source_image_folder, destination_image_folder_path):
        image_names = os.listdir(source_image_folder)
        num_of_indices_to_choose = round(len(image_names) * 0.3)

        random_image_indices = random.sample(range(len(image_names)), num_of_indices_to_choose)
        choosen_images = [(os.path.join(source_image_folder, image_names[i]), os.path.join(destination_image_folder_path, image_names[i]))
                        for i in random_image_indices]

        for source, destination in choosen_images:
            if source.split(".")[-1] in ["jpg", "png", "jpeg"]:
                shutil.copy(source, destination)
                all_files.append(source)



    def delete_moved_images(self):
        for image in all_files:
            os.remove(image)

        print(f"All choosen images from train set has been deleted")

    def _copy_and_delete(self):
        source_folders = os.listdir(SOURCE_DATA_PATH)

        for folder in source_folders:
            try:
                destination_image_folder_path = os.path.join(DESTINATION_DATAPATH, folder)
                self._does_exist(destination_image_folder_path)
            except Exception as e:
                raise Exception(f"Could not create validation folder {folder}")
            
            source_image_folder_path = os.path.join(SOURCE_DATA_PATH, folder)

            # if not os.path.isdir(source_image_folder_path):
            self._choose_file_index(source_image_folder_path, destination_image_folder_path)
        
        self.delete_moved_images()

    def __call__(self):
        self._does_exist()
        self._copy_and_delete()


if __name__ == '__main__':
    val_data_creation = ValidationData()
    val_data_creation()



