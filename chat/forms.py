from django import forms
from . import models
from captcha.fields import CaptchaField


class ContactForm(forms.Form):
	CITY = [
		['TP', "Taipei"],
		['TY', "Taoyuang"],
		['TC', "Taichung"],
		['TN', "Tainan"],
		['KS', "Kaohsiung"],
		['NA', "Others"],
	]
	user_name = forms.CharField(label='您的姓名', max_length=50, initial='Victor')
	user_city = forms.ChoiceField(label='居住城市', choices = CITY)
	user_school = forms.BooleanField(label='是否在學', required=False)
	user_email = forms.EmailField(label='電子郵件')
	user_message = forms.CharField(label='您的意見', widget=forms.Textarea)

class PostForm(forms.ModelForm):
	captcha = CaptchaField()
	class Meta:
		model = models.Post
		fields = ['mood', 'nickname', 'message', 'del_pass']

	def __init__(self, *args, **kwargs):
		super(PostForm, self).__init__(*args, **kwargs)
		self.fields['mood'].label = '現在心情'
		self.fields['nickname'].label = '你的暱稱'
		self.fields['message'].label = '心情留言'
		self.fields['del_pass'].label = '設定密碼'
		self.fields['captcha'].label = '不是機器人'

class Mail(forms.ModelForm):
	user_name = forms.CharField(label='您的姓名', max_length=50, initial='Victor')
	user_email = forms.EmailField(label='電子郵件')
	user_message = forms.CharField(label='您的意見', widget=forms.Textarea)

	class Meta:
		model = models.Mail
		fields = ['user_name', 'user_email', 'user_message']

	def __init__(self, *args, **kwargs):
		super(Mail, self).__init__(*args, **kwargs)
		self.fields['user_name'].label = '姓名'
		self.fields['user_email'].label = '信箱'
		self.fields['user_message'].label = '內容'
