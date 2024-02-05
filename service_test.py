import requests
import logging
import os
import time

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Use environment variable for URL or a direct string if not available
    url = os.getenv("API_URL", "http://localhost:8080/api/yolo8-coco-segmentation")

    # Define the payload with the source image, a random from web
    payload = {
        'src': 'https://t3.ftcdn.net/jpg/02/43/12/34/360_F_243123463_zTooub557xEWABDLk0jJklDyLSGl2jrr.jpg'
    }

    try:
        # Time the request
        start_time = time.time()
        
        # Make the POST request
        response = requests.post(url=url, json=payload)
        
        # Check if the response was successful
        if response.status_code == 200:
            logging.info(f"Time taken: {time.time() - start_time} seconds")
            logging.info(response.json())
        else:
            logging.error(f"Request failed with status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Request exception: {e}")

if __name__ == "__main__":
    main()
