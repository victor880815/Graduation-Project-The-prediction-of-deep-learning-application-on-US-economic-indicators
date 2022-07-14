from django.contrib import admin
from chat import models

# Register your models here.
class PostAdmin(admin.ModelAdmin):
	list_display=('nickname', 'message', 'enabled', 'pub_time')
	ordering=('-pub_time',)
admin.site.register(models.Mood)
admin.site.register(models.Post, PostAdmin)
admin.site.register(models.Mail)