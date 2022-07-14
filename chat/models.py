from django.db import models

# Create your models here.
		
class Mood(models.Model):
	status = models.CharField(max_length=10, null=False)
	def __str__(self):
		return self.status

class Post(models.Model):
	mood = models.ForeignKey('Mood', on_delete=models.CASCADE)
	nickname = models.CharField(max_length=10, default='不願意透漏身份的人')
	message = models.TextField(null=False)
	del_pass = models.CharField(max_length=10)
	bday = models.CharField(max_length=10, default='')
	pub_time = models.DateTimeField(auto_now=True)
	enabled = models.BooleanField(default=True)

	def __str__(self):
		return self.message

class Mail(models.Model):
	user_name = models.CharField(max_length=20)
	# user_city = models.CharField(max_length=20)
	user_email = models.CharField(max_length=20)
	user_message= models.TextField(null=False)

	def __str__(self):
		return self.user_message

