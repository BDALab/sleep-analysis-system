import logging
import os.path

import pytz
from django.contrib.admin.views.decorators import staff_member_required
from django.http import HttpResponse, FileResponse
from django.shortcuts import redirect
from django.shortcuts import render, get_object_or_404
from django.template import loader

from dashboard.export.export_hilev import export_all_features
from dashboard.logic.features_extraction.count_hilev import hilev
from dashboard.logic.preprocessing.preprocess_data import preprocess_all_data
from .export.export_actions import export_all, export_subject
from .export.export_hilev_avg import export_all_features_avg
from .export.export_hilev_clinic_data import export_all_features_avg_clinic
from .logic.machine_learning.learn import prepare_model
from .logic.machine_learning.predict import predict_all
from .logic.parkinson_analysis.train_classifier import train_parkinson_classifier
from .logic.reults_visualization.sleep_graph import create_graph
from .logic.sleep_diary.parse_metadata import parse_metadata
from .logic.sleep_diary.validate_sleep_wake import validate_sleep_wake
from .models import Subject, CsvData, SleepDiaryDay, RBDSQ, SleepNight

logger = logging.getLogger(__name__)


def index(request):
    template = loader.get_template('dashboard/index.html')
    context = {

    }
    return HttpResponse(template.render(context, request))


def subjects_page(request):
    if request.user.is_superuser or \
            request.user.groups.filter(name='researchers').exists() or \
            request.user.groups.filter(name='administrators').exists():
        subjects = Subject.objects.all().order_by("-creation_date")
    elif Subject.objects.filter(code=request.user.get_username()):
        subjects = [Subject.objects.filter(code=request.user.get_username()).first()]
        subjects.extend([s for s in Subject.objects.all().order_by("-creation_date") if s.is_test()])
    else:
        subjects = [s for s in Subject.objects.all().order_by("-creation_date") if s.is_test()]
    template = loader.get_template('dashboard/subjects.html')
    context = {
        'subjects': subjects,
    }
    return HttpResponse(template.render(context, request))


@staff_member_required
def detail(request, code):
    subject = get_object_or_404(Subject, code=code)
    if request.user.get_username() == code or \
            request.user.is_superuser or \
            request.user.groups.filter(name='researchers').exists() or \
            request.user.groups.filter(name='administrators').exists():
        logger.warning(f'User {request.user} access details of subject {code}')
        context = {
            'subject': subject,
        }

        sleep_nights = SleepNight.objects.filter(subject=subject)
        if sleep_nights.exists():
            data = []
            for night in sleep_nights:
                plot_div, fig = create_graph(night)
                data.append((plot_div, night.name_url, night.info, night.diary_day.info))
            context['data'] = data
        else:
            csv_data = CsvData.objects.filter(subject=subject)
            if csv_data.exists():
                data = []
                for d in csv_data:
                    plot_div, fig = create_graph(d)
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


@staff_member_required
def detail_action(request, code, action):
    subject = get_object_or_404(Subject, code=code)
    if request.user.get_username() == code or \
            request.user.is_superuser or \
            request.user.groups.filter(name='researchers').exists() or \
            request.user.groups.filter(name='administrators').exists():
        if action == 'export':
            export_subject(subject, request.user.get_full_name())
            if os.path.exists(f'{subject.export_name}.pdf'):
                return FileResponse(open(f'{subject.export_name}.pdf', 'rb'), as_attachment=True)
        else:
            return redirect('dashboard:index')


# Change to be async and show progress bar: https://buildwithdjango.com/blog/post/celery-progress-bars/
@staff_member_required
def utils(request, action=None):
    if not request.user.is_superuser:
        logger.warning(f'Blocked request to access utils site for user {request.user}')
        return redirect('dashboard:index')
    template = loader.get_template('dashboard/utils.html')

    if action == 'preprocess':
        logger.info(f'Preprocess data')
        if preprocess_all_data():
            logger.info('Preprocess OK')
            context = {
                'ok': 'Preprocess the data operation was successful'
            }
        else:
            logger.error('Preprocess failed')
            context = {
                'fail': 'Preprocess the data failed! Try to check all data.'
            }

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

    elif action == 'all':
        if preprocess_all_data() and predict_all() and hilev():
            logger.info('All operations OK')
            context = {
                'ok': 'All operations performed successfully'
            }
        else:
            logger.error('Some of the operations failed')
            context = {
                'fail': 'Some of the operations failed!'
            }

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
    elif action == 'hilev':
        logger.info('Calculate high level features')
        if hilev():
            logger.info('HiLev counted')
            context = {
                'ok': 'High level features counted'
            }
        else:
            logger.error('Count high level features end up with exception.')
            context = {
                'fail': 'Count HiLev failed!'}
    elif action == 'export':
        logger.info('Export subjects to pdf')
        if export_all(request.user.get_full_name()):
            logger.info('Export completed')
            context = {
                'ok': 'Export completed successfully'
            }
        else:
            logger.error('Failed to export subjects')
            context = {
                'fail': 'Export for all subjects failed!'}
    elif action == 'export_dataset':
        logger.info('Export dataset to excel')
        if export_all_features():
            logger.info('Export completed')
            context = {
                'ok': 'Export completed successfully'
            }
        else:
            logger.error('Failed to export dataset')
            context = {
                'fail': 'Export of features to dataset failed!'}
    elif action == 'export_dataset_avg':
        logger.info('Export dataset with average for each subject to excel')
        if export_all_features_avg():
            logger.info('Export completed')
            context = {
                'ok': 'Export completed successfully'
            }
        else:
            logger.error('Failed to export dataset')
            context = {
                'fail': 'Export of features to dataset failed!'}
    elif action == 'export_dataset_clinic':
        logger.info('Export dataset with average and clinic data for each subject to excel')
        if export_all_features_avg_clinic():
            logger.info('Export completed')
            context = {
                'ok': 'Export completed successfully'
            }
        else:
            logger.error('Failed to export dataset')
            context = {
                'fail': 'Export of features to dataset failed!'}

    elif action == 'parkinson_fnusa':
        logger.info('Train parkinson classifier')
        if train_parkinson_classifier():
            logger.info('Training completed')
            context = {
                'ok': 'Classifier trained successfully'
            }
        else:
            logger.error('Failed to train classifier')
            context = {
                'fail': 'Training of classifier failed!'}

    else:
        context = {}
    return HttpResponse(template.render(context, request))


def set_timezone(request):
    if request.method == 'POST':
        request.session['django_timezone'] = request.POST['timezone']
        return redirect('/')
    else:
        return render(request, 'dashboard/timezone.html', {'timezones': pytz.common_timezones})
