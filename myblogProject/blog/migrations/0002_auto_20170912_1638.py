# -*- coding: utf-8 -*-
# Generated by Django 1.11.5 on 2017-09-12 16:38
from __future__ import unicode_literals

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('blog', '0001_initial'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='entry',
            options={'verbose_name_plural': 'entries'},
        ),
        migrations.AddField(
            model_name='entry',
            name='created_date',
            field=models.DateTimeField(default=django.utils.timezone.now),
        ),
        migrations.AddField(
            model_name='entry',
            name='published_date',
            field=models.DateTimeField(blank=True, null=True),
        ),
    ]