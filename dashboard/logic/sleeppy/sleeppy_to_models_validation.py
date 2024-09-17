import logging

from dashboard.models import SleeppyData, SleepNight

logger = logging.getLogger(__name__)


def sleeppy_to_models_validation():
    nights_with_no_sleep_data = []
    logger.info("=== Sleeppy data objects with the same sleep nights ===")
    for sleep_night in SleepNight.objects.all():
        sleepy_models_with_same_nights = SleeppyData.objects.filter(sleep_night=sleep_night).all()
        if sleepy_models_with_same_nights.count() == 0:
            nights_with_no_sleep_data.append(sleep_night)

        if sleepy_models_with_same_nights.count() > 1:
            logger.info("###")
            logger.info(
                f'Sleep night: {sleep_night.date}, subject: {sleep_night.subject.code}, data: {sleep_night.data.filename} has more than one Sleeppy data objects:')
            for sleepy_model in sleepy_models_with_same_nights:
                logger.info(sleepy_model.info)
            logger.info("###")

    logger.info(
        f"=== Sleep nights without sleeppy data objects: {len(nights_with_no_sleep_data)} out of total {len(SleepNight.objects.all())} sleep nights ===")
    for sleep_night in nights_with_no_sleep_data:
        logger.info(
            f'Sleep night: {sleep_night.date}, subject: {sleep_night.subject.code}, data: {sleep_night.data.filename}')

    sleeppy_data_without_night = SleeppyData.objects.filter(sleep_night=None).all()
    logger.info(f"=== Sleeppy data objects without sleeppy night: {len(sleeppy_data_without_night)}===")
    for sleeppy_data in sleeppy_data_without_night:
        logger.info(
            f'Date: {sleeppy_data.date}, subject: {sleeppy_data.subject.code}, data: {sleeppy_data.data.filename}')

    sleeppy_data_with_100_efficiency = SleeppyData.objects.filter(se=100).all()
    logger.info(f"=== Sleeppy data objects with 100% sleep efficiency: {len(sleeppy_data_with_100_efficiency)}===")
    for sleeppy_data in sleeppy_data_with_100_efficiency:
        logger.info(
            f'Date: {sleeppy_data.date}, subject: {sleeppy_data.subject.code}, data: {sleeppy_data.data.filename}')
