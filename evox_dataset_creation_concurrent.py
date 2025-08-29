import concurrent.futures
import json
from botocore.config import Config
import requests
from requests.adapters import HTTPAdapter
import pandas as pd
import os
import time
from typing import Dict, List, Tuple, Optional
from threading import Semaphore
import logging
import re
from aws_helpers import helpers
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logging.basicConfig(level=logging.INFO, format='%(filename)s:%(funcName)s:%(lineno)d% - (levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VehicleProcessor:
    def __init__(self,
                 evox_api_key: str, 
                 bucket_name: str = 'signal-8-evox',
                 productID: int = 3,
                 productTypeID: int = 67,
                 image_workers: int = 10,
                 max_api_concurrency: int = 400):
        """
        Initialize the vehicle processor
        
        Args:
            bucket_name: S3 bucket name (default: signal-8-evox)
            max_api_concurrency: Maximum concurrent API calls (default 400)
        """
        self.bucket_name = bucket_name
        self.productID = productID
        self.productTypeID = productTypeID
        self.evox_api_key = evox_api_key
        self.session = requests.Session()
        # Create HTTPAdapter with increased pool size
        adapter = HTTPAdapter(
            pool_connections=50,    # Number of different hosts (default 10 is fine)
            pool_maxsize=500        # Max connections per host (increase from 10 to 100)
        )
        
        # Mount adapter for both HTTP and HTTPS
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        self.image_workers = image_workers
        self.max_api_concurrency = max_api_concurrency
        client_config = Config(
            max_pool_connections=500  # Increase from default 10 to 50
        )
        self.s3_client = helpers._get_s3_client(config=client_config)
        
        # Semaphore to limit concurrent API calls to 400
        self.api_semaphore = Semaphore(self.max_api_concurrency)
        
        # Results storage - now using array format
        self.processed_vehicles = []
        self.processed_count = 0
        self.failed_vehicles = []

    def load_vehicle_ids(self, excel_file_path: str, column_name: str = 'VIF #') -> List[str]:
        """
        Load vehicle IDs from Excel file
        
        Args:
            excel_file_path: Path to Excel file
            column_name: Column name containing vehicle IDs
        
        Returns:
            List of vehicle IDs
        """
        df = pd.read_excel(excel_file_path, sheet_name='Sheet1')
        logger.info(f"Loaded a total of {df.shape} from EXCEL")
        df = df[df['Exterior'] == 1]
        # df = df.iloc[:1000, :].copy()
        vehicle_ids = df[column_name].dropna().tolist()
        logger.info(f"Loaded {len(vehicle_ids)} vehicle IDs from Excel")
        return vehicle_ids

    def clean_filename_part(self, text: str) -> str:
        """
        Clean text for use in S3 key/filename
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text safe for filenames
        """
        if not text:
            return "Unknown"
        # Remove or replace characters that aren't filename-safe
        cleaned = re.sub(r'[<>:"/\\|?*]', '', str(text))
        cleaned = cleaned.replace(' ', '-')  # Keep spaces as spaces
        return cleaned.strip()

    def fetch_vehicle_data(self, vehicle_id: str) -> Dict:
        """
        Fetch vehicle details from API
        
        Args:
            vehicle_id: Unique vehicle identification number
            
        Returns:
            Dictionary containing vehicle details and image URLs
        """
        # Acquire semaphore to limit concurrent API calls
        self.api_semaphore.acquire()
        
        try:
            # Replace with your actual API endpoint
            url = f"https://api.evoximages.com/api/v1/vehicles/{vehicle_id}/products/{self.productID}/{self.productTypeID}?api_key={self.evox_api_key}"
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API call failed for vehicle {vehicle_id}: {e}")
            raise
        finally:
            # Always release the semaphore
            self.api_semaphore.release()

    def download_and_upload_image(self, image_url: str, vehicle_data: Dict, image_index: int) -> str:
        """
        Download image from URL and upload to S3 with specific naming convention
        
        Args:
            image_url: URL of the image to download
            vehicle_data: Vehicle details dictionary from API
            image_index: Index number of the image (1-36)
            
        Returns:
            S3 URI of uploaded image
        """
        try:
            # Download image with timeout
            response = self.session.get(image_url, timeout=30)
            response.raise_for_status()
            
            # Extract vehicle details for filename
            vifnum = vehicle_data.get('vifnum', 'Unknown')
            make = self.clean_filename_part(vehicle_data.get('make', 'Unknown'))
            model = self.clean_filename_part(vehicle_data.get('model', 'Unknown'))
            year = vehicle_data.get('year', 'Unknown')
            color = self.clean_filename_part(vehicle_data.get('color_simpletitle', 'Unknown'))
            
            # Generate S3 key with specified format: {vifnum}-{make}-{model}-{year}-{color}-{image_number}.jpeg
            s3_key = f"Images/{vifnum}-{make}-{model}-{year}-{color}-{image_index}.jpeg"
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=response.content,
                ContentType='image/jpeg'
            )
            
            # Return S3 URI
            s3_uri = f"s3://{self.bucket_name}/{s3_key}"
            logger.debug(f"Uploaded image {image_index} for vehicle {vifnum}")
            
            return s3_uri
            
        except Exception as e:
            logger.error(f"Failed to process image {image_index} for vehicle {vehicle_data.get('vifnum', 'Unknown')}: {e}")
            # Return empty string for failed uploads
            return ""

    def process_single_vehicle(self, vehicle_id: str) -> Optional[Dict]:
        """
        Process a single vehicle: fetch data, download/upload images, create JSON
        
        Args:
            vehicle_id: Vehicle identification number
            
        Returns:
            Processed vehicle data in the required format
        """
        try:
            # Step 1: Fetch vehicle data from API
            api_response = self.fetch_vehicle_data(vehicle_id)
            
            # Check if API response is successful
            if api_response.get('status') != 'success':
                logger.error(f"API returned error status for vehicle {vehicle_id}")
                return None
            
            # Extract vehicle details from the nested structure
            vehicle_data = api_response.get('vehicle', {})
            image_urls = api_response.get('urls', [])
            
            if not vehicle_data:
                logger.error(f"No vehicle data found for vehicle {vehicle_id}")
                return None
                
            if not image_urls:
                logger.warning(f"No images found for vehicle {vehicle_id}")
            
            # Step 2: Download and upload images concurrently (max 10 per vehicle)
            s3_uris = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.image_workers) as image_executor:
                # Submit all image download/upload tasks
                image_futures = {
                    image_executor.submit(self.download_and_upload_image, url, vehicle_data, idx + 1): (url, idx + 1)
                    for idx, url in enumerate(image_urls)
                }
                
                # Collect results as they complete and maintain order
                results = []
                for future in concurrent.futures.as_completed(image_futures):
                    url, image_index = image_futures[future]
                    try:
                        s3_uri = future.result()
                        if s3_uri:  # Only add successful uploads
                            results.append(s3_uri)
                    except Exception as e:
                        logger.error(f"Image processing failed for vehicle {vehicle_id}, image {image_index}: {e}")
                
                # Filter out None values and maintain order
                s3_uris = [s3uri for s3uri in results if s3uri is not None]
            
            # Step 3: Create vehicle object in required format
            vehicle_json = {
                "vifid": vehicle_data.get('vifnum'),
                "year": vehicle_data.get('year'),
                "make": vehicle_data.get('make'),
                "model": vehicle_data.get('model'),
                "trim": vehicle_data.get('trim'),
                "color": vehicle_data.get('color_simpletitle'),
                "car_type": vehicle_data.get('body'),
                "s3uris": s3_uris
            }

            # logger.info(f"Successfully processed vehicle {vehicle_data.get('vifnum')} with {len(s3_uris)} images")
            return vehicle_json
            
        except Exception as e:
            logger.error(f"Failed to process vehicle {vehicle_id}: {e}")
            self.failed_vehicles.append(vehicle_id)
            return None

    def process_all_vehicles(self, vehicle_ids: List[str]):
        """
        Process all vehicles concurrently with proper rate limiting
        
        Args:
            vehicle_ids: List of vehicle IDs to process
        """
        total_vehicles = len(vehicle_ids)
        logger.info(f"Starting to process {total_vehicles} vehicles with max {self.max_api_concurrency} concurrent API calls and {self.image_workers} image workers.")
        
        # Use ThreadPoolExecutor for concurrent processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_api_concurrency) as executor:
            # Submit all vehicle processing tasks
            future_to_vehicle = {
                executor.submit(self.process_single_vehicle, vehicle_id): vehicle_id 
                for vehicle_id in vehicle_ids
            }
            
            # Process completed tasks as they finish
            for future in concurrent.futures.as_completed(future_to_vehicle):
                vehicle_id = future_to_vehicle[future]
                
                try:
                    vehicle_data = future.result()
                    
                    if vehicle_data:
                        # Add to processed vehicles array
                        self.processed_vehicles.append(vehicle_data)
                        self.processed_count += 1
                        
                        # Log progress every 100 vehicles
                        if total_vehicles < 100:
                            logger.info(f"Progress: {self.processed_count}/{total_vehicles} vehicles processed")
                        elif self.processed_count % 100 == 0:
                            logger.info(f"Progress: {self.processed_count}/{total_vehicles} vehicles processed")
                    
                except Exception as e:
                    logger.error(f"Unexpected error processing vehicle {vehicle_id}: {e}")
                    self.failed_vehicles.append(vehicle_id)

    def save_results(self, output_file: str = 'dataset.json'):
        """
        Save processed data to JSON file in the required format
        
        Args:
            output_file: Output JSON file path
        """
        # Create the final JSON structure with "output" array
        final_json = {
            "output": self.processed_vehicles
        }
        
        # Save main results
        with open(output_file, 'w') as f:
            json.dump(final_json, f, indent=2)

        self.s3_client.put_object(Bucket=self.bucket_name,
                                  Key=output_file,
                                  Body=json.dumps(final_json, indent=2),
                                  ContentType='application/json')
        
        # Save failed vehicles list for retry
        if self.failed_vehicles:
            with open('failed_vehicles.json', 'w') as f:
                json.dump(self.failed_vehicles, f, indent=2)
        
        # Log summary
        total_processed = len(self.processed_vehicles)
        total_failed = len(self.failed_vehicles)
        total_images = sum(len(vehicle.get('s3uris', [])) for vehicle in self.processed_vehicles)
        
        logger.info(f"""
        Processing Summary:
        ==================
        Total vehicles processed: {total_processed}
        Total vehicles failed: {total_failed}
        Total images uploaded: {total_images}
        Results saved to: {output_file}
        Failed vehicles saved to: failed_vehicles.json
        """)

def main():
    """
    Main function to run the entire processing pipeline
    """
    # Configuration - UPDATE THESE VALUES
    EXCEL_FILE_PATH = 'VIF_list_American.xlsx'  # Replace with your Excel file path
    COLUMN_NAME = 'VIF #'  # Replace with your column name
    productID = 3
    productTypeID = 67
    EVOX_API_KEY = os.getenv("EVOX_API_KEY", None)

    if EVOX_API_KEY is None:
        raise ValueError("EVOX api missing or incorrect, check.")
    
    # Initialize processor with signal-8-evox bucket
    processor = VehicleProcessor(evox_api_key = EVOX_API_KEY,
                                 bucket_name='signal-8-evox',
                                 productID=productID,
                                 productTypeID=productTypeID,
                                 max_api_concurrency=50,
                                 image_workers=10)
    
    try:
        # Step 1: Load vehicle IDs from Excel
        vehicle_ids = processor.load_vehicle_ids(EXCEL_FILE_PATH, COLUMN_NAME)
        
        # Step 2: Process all vehicles
        start_time = time.time()
        processor.process_all_vehicles(vehicle_ids)
        end_time = time.time()
        
        # Step 3: Save results
        processor.save_results()
        
        logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise

if __name__ == "__main__":
    main()
