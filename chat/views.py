from django.shortcuts import render
from django.http import HttpResponseRedirect
from chat import models
from .import forms
from django.contrib.sessions.models import Session
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from datetime import datetime


# Create your views here.
@login_required(login_url='signin')
def chat_post(request, pid=None, del_pass=None):
	posts = models.Post.objects.filter(enabled = True).order_by('-pub_time')[:30]
	moods = models.Mood.objects.all()
	now = datetime.now()

	try:
		user_id = request.GET["user_id"]
		user_pass = request.GET["user_pass"]
		user_post = request.GET["user_post"]
		user_mood = request.GET["mood"]
	except:
		user_id = None
		message = '欄位皆必填'

	if del_pass and pid:
		try:
			post = models.Post.objects.get(id=pid)
		except:
			post = None
		if post:
			if post.del_pass == del_pass:
				post.delete()
				message= "刪除成功"
			else:
				message = "密碼錯誤"

	elif user_id!=None:
		m = models.Mood.objects.get(status = user_mood)
		post = models.Post.objects.create(mood = m, nickname = user_id, del_pass = user_pass, message = user_post)
		post.save()
		message = '儲存成功'

	return render(request, "chat.html", locals())

def contact(request):
	now = datetime.now()

	if request.method == "POST":
		form = forms.Mail(request.POST)
		if form.is_valid():
			user_name = form.cleaned_data['user_name']
			# user_city = form.cleaned_data['user_city']
			user_email = form.cleaned_data['user_email']
			user_message = form.cleaned_data['user_message']
			message = "感謝來信"
			form.save()
		else:
			message = "資料不正確"
	else:
		form = forms.Mail()	
	return render(request, 'chat/contact.html', locals())