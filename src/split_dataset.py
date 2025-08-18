import os


SOURCE_DATA_PATH = "/mnt/A4F0E4F6F0E4D01A/Shams Iqbal/VS code/Kaggle/Datasets/CIFAR-100-dataset/test (Copy)"
DESTINATION_DATAPATH = "/mnt/A4F0E4F6F0E4D01A/Shams Iqbal/VS code/Kaggle/Datasets/CIFAR-100-dataset/val"

class ValidationData:
    def __init__(self):
        self.does_exist()

    
    def does_exist():
        try:
            if not os.path.exists(DESTINATION_DATAPATH):
                os.mkdir(DESTINATION_DATAPATH)
        except:
            raise Exception("Validation folder could not be created")


    def _():
        source_folders = os.listdir(SOURCE_DATA_PATH)

        for folder in source_folders:
            try:
                os.mkdir(os.path.join(DESTINATION_DATAPATH, folder))
            except Exception as e:
                raise Exception(f"Could not create validation folder {folder}")
            
            source_image_folder_path = os.path.join(SOURCE_DATA_PATH, folder)

            if os.path.isdir(source_image_folder_path):
                images = os.path.join()


