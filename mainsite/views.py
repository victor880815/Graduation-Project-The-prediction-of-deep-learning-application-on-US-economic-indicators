from django.http import HttpResponse
from datetime import datetime
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .scrapy import First_unemployed, Continue_unemployed, ML_Home, Model_Training
import time
from datetime import datetime

def machinelearning(request):
	source1 = request.POST.get('source1')
	source2 = request.POST.get('source2')
	first = request.POST.get('first')
	continued = request.POST.get('continued')
	ddate1 = request.POST.get('ddate1')

	ml = ML_Home(source1, source2, first, continued, ddate1)
	context_ml = {"infos_ml":ml.ML()}

	return render(request,'ml.html', context_ml)


def model(request):
	source1 = request.POST.get('source1')
	source2 = request.POST.get('source2')
	first = request.POST.get('first')
	continued = request.POST.get('continued')
	keyword = request.POST.get('keyword')

	mt = Model_Training(source1, source2, first, continued, keyword)
	context_mt = {"infos_mt":mt.Model()}

	return render(request,'model.html', context_mt)


def tv(request,tvno = 0):
	tv_list = [{'name':'財經週末趴','tvcode':'HQijPqO37nw'},
				{'name':'財經一路發','tvcode':'V2fwGFONkww'},
				{'name':'錢線百分百','tvcode':'TJO6rdMqdXo'},
				{'name':'錢線煉金術','tvcode':'5jwFjiXBt9Y'},]
	now = datetime.now()
	tvno = tvno
	tv = tv_list[tvno]

	return render(request,'tv.html',locals())


def f_unemployed(request):
	
	
	ddate1 = request.POST.get('ddate1')
	ddate2 = request.POST.get('ddate2')

	first_unemployed = First_unemployed(ddate1,ddate2)
	context={"infos":first_unemployed.scrape()}

	return render(request, "f_unemployed.html", context)


def c_unemployed(request):
	
	ddate1 = request.POST.get('ddate1')
	ddate2 = request.POST.get('ddate2')

	continue_unemployed = Continue_unemployed(ddate1,ddate2)
	context2={"infos2":continue_unemployed.scrape()}
	
	return render(request, "c_unemployed.html", context2)
