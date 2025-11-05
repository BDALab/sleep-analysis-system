import logging
import shutil
from datetime import datetime
from pathlib import Path

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
            stem = Path(csv_data.data.path).stem
            results_dir = Path(csv_data.sleeppy_dir).resolve() / stem / "results"
            expected_outputs = [
                results_dir / f"{stem}_major_rest_periods.csv",
                results_dir / "sleep_endpoints_summary.csv",
            ]
            if not force and all(p.exists() and p.stat().st_size > 0 for p in expected_outputs):
                logger.info(
                    f'SleepPy results already present for {csv_data.filename} at {results_dir}, skipping. '
                    f'Use force=True to regenerate.'
                )
                return True

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
            if any(not p.exists() or p.stat().st_size == 0 for p in expected_outputs):
                logger.error(
                    f'SleepPy outputs missing or empty for {csv_data.filename} (expected at {results_dir}).'
                )
                return False
        except Exception as e:
            logger.error(f'{csv_data.filename} failed due to {e}')
            return False
        end = datetime.now()
        logger.info(f'SleepPy for {csv_data.filename} made in {end - start}')
        return True


def sleeppy_clean():
    data = CsvData.objects.filter(training_data=False).all()
    for d in data:
        src_name = Path(d.data.path).stem
        sub_dir = Path(d.sleeppy_dir).resolve() / src_name
        if sub_dir.exists():
            results_dir = sub_dir / "results"
            if not results_dir.exists():
                shutil.rmtree(sub_dir)
                logger.info(f"{sub_dir} deleted")
    return True
