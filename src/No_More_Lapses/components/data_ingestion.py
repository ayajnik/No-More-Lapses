import os
import zipfile
import gdown
from src.No_More_Lapses import logger
from src.No_More_Lapses.utils.common import get_size
from src.No_More_Lapses.entity.config_entity import (DataIngestionConfig)



class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    
    def download_file(self) -> str:
        '''
        Fetch data from the url
        '''
        try: 
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            # Extract the file ID from the Google Drive URL
            if "drive.google.com" in dataset_url:
                if "/file/d/" in dataset_url:
                    # Format: https://drive.google.com/file/d/{FILE_ID}/view
                    file_id = dataset_url.split("/file/d/")[1].split("/")[0]
                elif "id=" in dataset_url:
                    # Format: https://drive.google.com/uc?id={FILE_ID}
                    file_id = dataset_url.split("id=")[1].split("&")[0]
                else:
                    # Try to extract from your current approach as fallback
                    file_id = dataset_url.split("/")[-2]
                
                logger.info(f"Extracted file_id: {file_id}")
                
                # Use gdown's direct ID download method
                url = f"https://drive.google.com/uc?id={file_id}"
                #gdown.download(id=url, output=str(zip_download_dir), quiet=False)
            else:
                # For non-Google Drive URLs, you might want to use a different method
                import requests
                response = requests.get(dataset_url)
                with open(zip_download_dir, 'wb') as f:
                    f.write(response.content)
            
            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")
            return zip_download_dir
        except Exception as e:
            logger.error(f"Error downloading file: {str(e)}")
            raise e
        
    

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)