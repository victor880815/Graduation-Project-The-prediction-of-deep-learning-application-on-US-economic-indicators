# Generated by Django 3.2.12 on 2022-05-29 11:36

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('mainsite', '0003_delete_question'),
    ]

    operations = [
        migrations.DeleteModel(
            name='Post',
        ),
    ]