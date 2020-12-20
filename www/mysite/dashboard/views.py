import logging

from django.contrib.admin.views.decorators import staff_member_required
from django.http import HttpResponse
from django.shortcuts import redirect
from django.shortcuts import render, get_object_or_404
from django.template import loader

from dashboard.logic.features_extraction.extract_features import extract_features_all
from dashboard.logic.preprocessing.preprocess_data import preprocess_all_data
from .logic.machine_learning.learn import prepare_model
from .logic.machine_learning.predict import predict_all
from .logic.reults_visualization.sleep_graph import create_graph
from .logic.sleep_diary.parse_metadata import parse_metadata
from .logic.sleep_diary.validate_sleep_wake import validate_sleep_wake
from .logic.utils_check import check_all_data, check_extracted_features, check_cached_data, check_model, \
    check_predicted_features
from .logic.utils_delete import delete_all_data, delete_cached_data, delete_extracted_features, delete_model, \
    delete_predicted_data
from .models import Subject, CsvData, SleepDiaryDay, RBDSQ

logger = logging.getLogger(__name__)


def index(request):
    template = loader.get_template('dashboard/index.html')
    context = {

    }
    return HttpResponse(template.render(context, request))


def subjects_page(request):
    if request.user.is_superuser:
        subjects = Subject.objects.all()
    elif Subject.objects.filter(code=request.user.get_username()):
        subjects = Subject.objects.filter(code=request.user.get_username())
        test = [s for s in Subject.objects.all() if s.is_test()]
        subjects = subjects.union(test)
    else:
        subjects = [s for s in Subject.objects.all() if s.is_test()]
    template = loader.get_template('dashboard/subjects.html')
    context = {
        'subjects': subjects,
    }
    return HttpResponse(template.render(context, request))


@staff_member_required
def detail(request, code):
    subject = get_object_or_404(Subject, code=code)
    if request.user.get_username() == code or request.user.is_superuser:
        logger.warning(f'User {request.user} access details of subject {code}')
        context = {
            'subject': subject,
        }

        csv_data = CsvData.objects.filter(subject=subject)
        if csv_data.exists():
            data = []
            for d in csv_data:
                plot_div = create_graph(d)
                data.append((plot_div, d.excel_prediction_url))
            context['data'] = data

        sleep_diary = SleepDiaryDay.objects.filter(subject=subject)
        if sleep_diary.exists():
            context['diary'] = sleep_diary

        rbdsq = RBDSQ.objects.filter(subject=subject)
        if rbdsq.exists():
            score = rbdsq.latest('creation_date').score()
            context['rbdsq'] = score

        return render(request, 'dashboard/detail.html', context)
    else:
        logger.warning(f'Blocked request to access subject {code} data for user {request.user}')
        return redirect('dashboard:index')


# Change to be async and show progress bar: https://buildwithdjango.com/blog/post/celery-progress-bars/
@staff_member_required
def utils(request, action=None):
    if not request.user.is_superuser:
        logger.warning(f'Blocked request to access utils site for user {request.user}')
        return redirect('dashboard:index')
    template = loader.get_template('dashboard/utils.html')

    # Cache
    if action == 'cache':
        logger.info(f'Cache data')
        if preprocess_all_data():
            logger.info('Cache OK')
            context = {
                'ok': 'Cache the data operation was successful'
            }
        else:
            logger.error('Cache failed')
            context = {
                'fail': 'Cache the data failed! Try to check all data.'
            }
    elif action == 'cache_check':
        logger.info(f'Check cached data')
        if check_cached_data():
            logger.info('Cached data OK')
            context = {
                'ok': 'Cached data OK'
            }
        else:
            logger.error('Cached data are broken')
            context = {
                'fail': 'Cached data are broken! Try to clean the data and repeat cache.'
            }
    elif action == 'cache_clean':
        logger.warning(f'Clean up cached data performed by user {request.user}')
        if delete_cached_data():
            logger.info('Clean up OK')
            context = {
                'ok': 'Data clean up successful'
            }
        else:
            logger.error('Clean up failed')
            context = {
                'fail': 'Data clean up failed!'
            }

    # Features
    elif action == 'features':
        logger.info(f'Extracting features')
        if extract_features_all():
            logger.info('Features extraction OK')
            context = {
                'ok': 'Features extraction successful'
            }
        else:
            logger.error('Features extraction failed')
            context = {
                'fail': 'Extraction of features failed! Try to check data consistency and preprocess data first.'
            }
    elif action == 'features_check':
        logger.info(f'Checking extracted features')
        if check_extracted_features():
            logger.info('Extracted features OK')
            context = {
                'ok': 'Extracted features are consistent'
            }
        else:
            logger.error('Extracted features consistency failed')
            context = {
                'fail': 'Extracted features consistency check failed! '
                        'Try to clean the data and repeat preprocess and extraction '
            }
    elif action == 'features_clean':
        logger.warning(f'Clean up features extracted performed by user {request.user}')
        if delete_extracted_features():
            logger.info('Clean up OK')
            context = {
                'ok': 'Extracted features clean up successful'
            }
        else:
            logger.error('Clean up failed')
            context = {
                'fail': 'Extracted features clean up failed!'
            }

    # Train
    elif action == 'learn':
        logger.info(f'Training ML model')
        if prepare_model():
            logger.info('Model training OK')
            context = {
                'ok': 'Model training was successful'
            }
        else:
            logger.error('Training ML failed')
            context = {
                'fail': 'Training model failed! Try to check data consistency first.'
            }
    elif action == 'learn_check':
        logger.info(f'Check trained ML model')
        if check_model():
            logger.info('Model trained OK')
            context = {
                'ok': 'Trained model cached and predicting properly'
            }
        else:
            logger.error('Check ML model failed')
            context = {
                'fail': 'Trained model is not cached or predicting improperly!'
            }
    elif action == 'learn_clean':
        logger.info(f'Delete trained ML model')
        if delete_model():
            logger.info('Model deleted')
            context = {
                'ok': 'Trained model deleted'
            }
        else:
            logger.error('Delete the model failed')
            context = {
                'fail': 'Trained model deletion failed!'
            }

    # Predict
    elif action == 'predict':
        logger.info(f'Caching data for prediction')
        if predict_all():
            logger.info('Prediction cached OK')
            context = {
                'ok': 'Cache prediction data was successful'
            }
        else:
            logger.error('Prediction cache failed')
            context = {
                'fail': 'Cache prediction data failed!'
            }
    elif action == 'predict_check':
        logger.info(f'Check predicted data')
        if check_predicted_features():
            logger.info('Predicted data ok')
            context = {
                'ok': 'Prediction data cached properly'
            }
        else:
            logger.error('Predicted data check fail')
            context = {
                'fail': 'Cached prediction data are broken!'
            }
    elif action == 'predict_clean':
        logger.info(f'Delete cached prediction data')
        if delete_predicted_data():
            logger.info('Cached predictions deleted')
            context = {
                'ok': 'Cached predictions deleted'
            }
        else:
            logger.error('Delete cached predictions failed')
            context = {
                'fail': 'Delete cached predictions failed!'
            }

    # All
    elif action == 'all':
        if preprocess_all_data() and extract_features_all() and prepare_model() and predict_all():
            logger.info('All operations OK')
            context = {
                'ok': 'All operations performed successfully'
            }
        else:
            logger.error('Some of the operations failed')
            context = {
                'fail': 'Some of the operations failed!'
            }
    elif action == 'all_check':
        logger.info(f'Check data consistency')
        if check_all_data():
            logger.info('Data consistency OK')
            context = {
                'ok': 'Data consistency is correct'
            }
        else:
            logger.error('Data consistency failed')
            context = {
                'fail': 'Data consistency was broken. You can try to clean up all data and do all the operations again.'
            }
    elif action == 'all_clean':
        logger.warning(f'Clean up all data performed by user {request.user}')
        if delete_all_data():
            logger.info('Clean up OK')
            context = {
                'ok': 'All data clean up successful'
            }
        else:
            logger.error('Clean up failed')
            context = {
                'fail': 'All data clean up failed!'}

    elif action == 'metadata':
        logger.info('Read metadata')
        if parse_metadata():
            logger.info('All metadata parsed')
            context = {
                'ok': 'All metadata parsed correctly'
            }
        else:
            logger.error('Metadata parsing failed for some entry.')
            context = {
                'fail': 'Some metadata were not parsed!'}
    elif action == 'validate_sleep_wake':
        logger.info('Validate sleep wake')
        if validate_sleep_wake():
            logger.info('Validation completed')
            context = {
                'ok': 'Validation completed, see log'
            }
        else:
            logger.error('Validation failed with an exception.')
            context = {
                'fail': 'Validation failed!'}

    else:
        context = {}

    return HttpResponse(template.render(context, request))
