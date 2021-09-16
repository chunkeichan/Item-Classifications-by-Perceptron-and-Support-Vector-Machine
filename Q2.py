#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
       Q2.py
"""
import glob 
fcf=sorted(glob.glob('data/*.txt'))
file=open(fcf[0],'r')
text1=file.read()
file.close()

file2=open(fcf[50],'r')
features=file2.read().split()
feature1=features[0]
occurence=text1.count(feature1)
