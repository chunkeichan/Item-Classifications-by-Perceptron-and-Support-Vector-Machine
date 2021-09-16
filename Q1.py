#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
       Q1.py
"""
import glob 
fcf=sorted(glob.glob('data/*.txt'))
file=open(fcf[0],'r')
text1=file.read()
file.close()
