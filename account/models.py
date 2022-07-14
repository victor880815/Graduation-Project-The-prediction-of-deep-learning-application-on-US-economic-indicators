from django.db import models

class Account(models.Model):
	username = models.CharField(max_length=30, unique=True)
	email = models.CharField(max_length=30, unique=True)

	def __str__(self):
		return self.username
