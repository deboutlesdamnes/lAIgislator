import os
import csv
from pathlib import Path
from bs4 import BeautifulSoup
from time import sleep
from tqdm import tqdm
import re

'''
input_path = filedialog.askdirectory()
output_path = Path(input_path).parent / "billtext"
folder_name = os.path.splitext(os.path.basename(input_path))[0]
'''

#input_file = filedialog.askopenfilename(initialdir=input_path)
#input_name = os.path.splitext(os.path.basename(input_file))[0]


def add_space(tag):
    if tag is not None:
        if tag.string:
            tag.string = ' ' + tag.string + ' '
        for child in tag.find_all(recursive=False):
            if child.name:
                add_space(child)

def get_title(soup):
    try:
        title = soup.find('dc:title').text
    except: 
        title = ''
    return title

def get_status(soup):
    try:
        bill = soup.find('bill').text
        status = bill['bill-stage']
        status = status.replace('-', ' ')
    except: 
        status = ''
    return status

def get_chamber(soup):
    try:
        chamber = soup.find('dc:publisher').text
        if chamber.startswith('U.S. '):
            chamber = chamber[5:]
        else:
            chamber = chamber
    except: 
        chamber = ''
    return chamber

def get_date(soup):
    try:
        date = soup.find('dc:date').text
    except: 
        date = ''
    return date

def get_billtext(soup):
    legis_body = soup.find('legis-body')
    #add_space(legis_body)
    if legis_body is not None:
        for entry in legis_body.find_all("toc-entry"):
            text = entry.get_text()
            try:
                no_weirdspaces_text = re.sub("\u2002", " ", text)
                entry.string.replaceWith(no_weirdspaces_text)
            except :
                text = text
        for enum in legis_body.find_all("enum"):
            enum.insert_after(" ")
        for paragraph in legis_body.find_all("paragraph"):
            paragraph.insert_before("\n")
        for header in legis_body.find_all("header"):
            header.insert_after("\n")
        return legis_body.get_text()
    else: return ''


def write_to_folder(input_path, output_path):
    for input_file in tqdm(os.listdir(input_path), colour="GREEN"):
        filename = os.path.splitext(input_file)[0]
        if os.path.isfile(os.path.join(input_path, input_file)):
            with open(os.path.join(input_path, input_file), 'r', encoding="utf-8") as input:
                soup = BeautifulSoup(input, features="xml")
                output_file = output_path.joinpath(filename + ".txt")
        # Process the file
            with open(output_file, 'w', encoding="utf-8") as output_file:
                output_file.write(get_billtext(filename, soup))

#print("Now writing billtexts...")
#write_to_folder()
