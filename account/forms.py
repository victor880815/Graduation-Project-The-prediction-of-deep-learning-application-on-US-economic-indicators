from django.forms import ModelForm
from django.contrib.auth.forms import UserCreationForm
from django import forms
from django.contrib.auth.models import User
from captcha.fields import CaptchaField


class CreateUserForm(UserCreationForm):
	username = forms.CharField(
        label="帳號",
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
	email = forms.EmailField(
		label="電子郵件",
		widget=forms.EmailInput(attrs={'class': 'form-control'})
	)
	password1 = forms.CharField(
	    label="密碼",
	    widget=forms.PasswordInput(attrs={'class': 'form-control'})
	)
	password2 = forms.CharField(
	    label="密碼確認",
	    widget=forms.PasswordInput(attrs={'class': 'form-control'})
	)
	captcha=CaptchaField()
	class Meta:
		model = User
		fields = ['username', 'email', 'password1', 'password2']
	def __init__(self, *args, **kwargs):
		super(CreateUserForm, self).__init__(*args, **kwargs)
		self.fields['captcha'].label = '不是機器人'

class RegistrationForm(forms.Form):
	username = forms.CharField(
		label = '帳號',
		widget = forms.TextInput(attrs={'class': 'form-control'})
	)
	password = forms.CharField(
		label = '密碼',
		widget = forms.PasswordInput(attrs={'class': 'form-control'})
	)