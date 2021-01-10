from django.contrib import admin
from nested_inline.admin import NestedTabularInline, NestedModelAdmin

from .models import Subject, CsvData, PsData, SleepDiaryDay, WakeInterval, RBDSQ


# Register your models here.


class PsDataInline(admin.TabularInline):
    inlines = ''
    model = PsData
    extra = 1


class CsvDataInline(admin.TabularInline):
    inlines = ''
    model = CsvData
    extra = 1


class WakeIntervalInline(NestedTabularInline):
    model = WakeInterval
    extra = 1
    fk_name = 'sleep_diary_day'


class RBDSQInline(admin.StackedInline):
    inlines = ''
    model = RBDSQ
    extra = 1


class SleepDiaryInline(NestedTabularInline):
    model = SleepDiaryDay
    fk_name = 'subject'
    extra = 7
    inlines = [WakeIntervalInline]


@admin.register(Subject)
class SubjectAdmin(NestedModelAdmin):
    fieldsets = [
        (None, {'fields': ['code']}),
        ('Subject info', {'fields': ['age', 'sex']}),
        ('Diagnosis', {'fields': ['sleep_disorder', 'diagnosis']})
    ]
    inlines = [CsvDataInline, RBDSQInline, SleepDiaryInline]


@admin.register(CsvData)
class CsvAdmin(admin.ModelAdmin):
    inlines = [PsDataInline]


@admin.register(SleepDiaryDay)
class SleepDiaryAdmin(admin.ModelAdmin):
    fieldsets = [
        ('General info', {'fields': ['subject', 'date']}),
        ('Lifestyle info', {
            'fields': [('day_sleep_count', 'day_sleep_time'), ('alcohol_count', 'alcohol_time'),
                       ('caffeine_count', 'caffeine_time'), 'sleeping_pill']
        }),
        ('Sleep info', {
            'fields': ['sleep_time', 'sleep_duration', 'wake_count', 'wake_time', 'get_up_time']
        }),
        ('Sleep quality', {'fields': ['sleep_quality', 'rest_quality']})
    ]
    inlines = [WakeIntervalInline]


admin.site.register(PsData)
admin.site.register(RBDSQ)

admin.sites.AdminSite.site_header = 'GENEActiv data processing administration'
admin.sites.AdminSite.site_title = 'GENEActiv data processing administration'
