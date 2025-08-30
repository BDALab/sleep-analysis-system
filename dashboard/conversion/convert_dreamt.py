import logging
from pathlib import Path

import pandas as pd

DREAMT_64HZ_FOLDER = "media/dreamt/data_64Hz"
DREAMT_64HZ_PROCESSED_FOLDER = "media/dreamt/data_64Hz_processed"
logger = logging.getLogger(__name__)


def convert_64hz_dreamt() -> bool:
    """
        Loads and processes all CSV files from a specified folder.

        Args:
            folder_path: The string path to the folder containing data files.

        Returns:
            True if all files were processed successfully, False otherwise.
        """
    # 1. Define the data and output directories
    data_dir = Path(DREAMT_64HZ_FOLDER)
    output_dir = Path(DREAMT_64HZ_PROCESSED_FOLDER)

    # 2. Check if the directory exists
    if not data_dir.is_dir():
        logging.error(f"Directory not found: {data_dir}")
        return False
    # And create the output directory if it doesn't exist ---
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Output will be saved to: {output_dir}")
    except OSError as e:
        logging.error(f"Could not create output directory {output_dir}. Error: {e}")
        return False

    # 3. Find all files in the directory.
    # We'll use .glob('*.csv') to find all files ending with .csv
    # You can change this to '*.txt' or '*.*' for other file types.
    files_to_process = list(data_dir.glob('*.csv'))

    if not files_to_process:
        logging.warning(f"No CSV files found in directory: {data_dir}")
        return False

    logging.info(f"Found {len(files_to_process)} files to process.")
    all_successful = True

    # 4. Loop through each file path
    for file_path in files_to_process:
        logging.info(f"--- Processing file: {file_path.name} ---")
        columns_to_export = []
        try:
            # 5. Load the data from the file into a pandas DataFrame
            # This assumes the file is a CSV. For other formats, you might use:
            # pd.read_excel(), pd.read_json(), etc.
            df = pd.read_csv(file_path)
            logging.info(f"  - Data loaded successfully. Shape: {df.shape}")

            # Process timestamp if it exists
            if 'TIMESTAMP' in df.columns:
                fix_start_time = pd.Timestamp("2025-03-13 19:00:00.000")
                df['processed_time'] = fix_start_time + pd.to_timedelta(df['TIMESTAMP'], unit='s')
                columns_to_export.append('processed_time')

            # Convert acceleration columns from 1/64g to g
            accel_cols = ['ACC_X', 'ACC_Y', 'ACC_Z']
            if all(col in df.columns for col in accel_cols):
                logging.info("  - Converting acceleration columns to g-units.")

                # Create new column names
                new_accel_cols = [f"{col}_g" for col in accel_cols]

                # Perform the conversion efficiently using vectorization
                df[new_accel_cols] = df[accel_cols] / 64.0

                logging.info("  - Conversion complete. Verifying new columns:")
                # Display the original and newly converted columns
                print(df[['ACC_X', 'ACC_X_g', 'ACC_Y', 'ACC_Y_g', 'ACC_Z', 'ACC_Z_g']].head())
                columns_to_export.extend(new_accel_cols)
            else:
                logging.warning(
                    f"  - One or more acceleration columns {accel_cols} not found. Skipping g-unit conversion.")

            if 'TEMP' in df.columns:
                columns_to_export.append("TEMP")

            if 'Sleep_Stage' in df.columns:
                logging.info("  - Converting 'Sleep_Stage' to binary sleep/wake column (0=Wake, 1=Sleep).")
                # Define the mapping: 1 for any sleep stage, 0 for wake.
                # This assumes 'P' is a sleep stage (like REM/Paradoxical sleep).
                sleep_map = {
                    'W': 0,  # Wake - wake
                    'P': 0,  # Wake - preparation
                    'N1': 1,  # Sleep - NREM 1
                    'N2': 1,  # Sleep - NREM 2
                    'N3': 1,  # Sleep - NREM 3
                    'R': 1,  # Sleep - REM
                }
                # Create a new column 'sleep_binary' using the map
                df['sleep_binary'] = df['Sleep_Stage'].map(sleep_map)
                columns_to_export.append('sleep_binary')
                logging.info("  - 'sleep_binary' column created.")

            # --- EXPORT THE PROCESSED DATAFRAME ---
            # Create a new filename for the processed file
            output_filename = f"{file_path.stem}_processed.csv"
            output_path = output_dir / output_filename

            # Save the DataFrame to a new CSV file, without the pandas index
            df.to_csv(output_path, columns=columns_to_export, index=False)
            logging.info(f"  - Successfully saved processed file to: {output_path}")
            # --- END OF EXPORT LOGIC ---

            # --- END OF PROCESSING LOGIC ---

        except pd.errors.EmptyDataError:
            logging.warning(f"  - Skipping empty file: {file_path.name}")
        except Exception as e:
            logging.error(f"  - Failed to process file {file_path.name}. Error: {e}")
            all_successful = False
            continue  # Move to the next file

    return all_successful
