from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.auth.forms import UserCreationForm
from .forms import CreateUserForm, RegistrationForm
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from .models import *


def signin(request):
	form = RegistrationForm()

	if request.method == 'POST':
		username = request.POST.get('username')
		password = request.POST.get('password')

		user = authenticate(request, username=username, password=password)

		if user is not None:
			login(request, user)
			return redirect('home')
		else:
			messages.info(request, '帳號或密碼錯誤')

	context = {'form':form}
	
	return render(request, 'signin.html', locals())


def signup(request):
	if request.user.is_authenticated:

		return redirect('home')

	else:
		form = CreateUserForm()
		if request.method == 'POST':
			form = CreateUserForm(request.POST)
			if form.is_valid():
				form.save()
				user = form.cleaned_data.get('username')
				messages.success(request, '已創立帳號'+ user)
				return redirect('/signin.html')
				
		context = {'form':form}

		return render(request, 'signup.html', locals())


def logoutUser(request):
	logout(request)

	return redirect('home')


def needhelp(request):

	return render(request, 'help.html', locals())
