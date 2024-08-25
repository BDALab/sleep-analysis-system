import logging
import os.path
import shutil
from datetime import datetime

from dashboard.logic.sleeppy.sleeppy_core import SleepPy
from dashboard.models import CsvData

logger = logging.getLogger(__name__)


def sleeppy_all():
    start = datetime.now()
    data = CsvData.objects.filter(training_data=False).all()
    logger.info(f'{len(data)} csv data objects will be used for Sleeppy experiment')
    result = True
    for d in data:
        if not sleeppy(d):
            result = False
    end = datetime.now()
    logger.info(f'Sleeppy experiment of all the {len(data)} data took {end - start}')
    return result


def sleeppy(csv_data, force=False):
    if isinstance(csv_data, CsvData):
        start = datetime.now()
        logger.info(f'SleepPy for {csv_data.filename}')
        try:
            if csv_data.end_date is None:
                logger.info("Working without end date...")
                sleepy = SleepPy(
                    input_file=csv_data.data.path,
                    results_directory=csv_data.sleeppy_dir,
                    sampling_frequency=25,
                    verbose=True
                )
            else:
                logger.info(f'Working with end date {csv_data.end_date}')
                sleepy = SleepPy(
                    input_file=csv_data.data.path,
                    results_directory=csv_data.sleeppy_dir,
                    sampling_frequency=25,
                    verbose=True,
                    stop_time=csv_data.end_date.strftime("%Y-%m-%d %H:%M:%S:%f")
                )
            sleepy.run_config = 0
            sleepy.run()
        except Exception as e:
            logger.error(f'{csv_data.filename} failed due to {e}')
            return False
        end = datetime.now()
        logger.info(f'SleepPy for {csv_data.filename} made in {end - start}')
        return True


def sleeppy_clean():
    data = CsvData.objects.filter(training_data=False).all()
    for d in data:
        src_name = d.data.path.split("/")[-1][0:-4]
        sub_dir = (d.sleeppy_dir + "/" + src_name)
        if os.path.exists(sub_dir):
            results_dir = sub_dir + "/results"
            if not os.path.exists(results_dir):
                shutil.rmtree(sub_dir)
                logger.info(f"{sub_dir} deleted")
    return True
