import requests
from bs4 import BeautifulSoup
import io
import os
import codecs

def getShortText(websiteUrl, textLength):
    page = requests.get(websiteUrl)

    #print(page.content)

    soup = BeautifulSoup(page.content, "html.parser")
        
    results = soup.find(id="block-ewcms-theme-main-page-content")

    list_headings = results.find_all("h2", class_="ecl-u-type-heading-2")

    list_texts = results.find_all("div", class_="ecl-u-mb-2xl")

    list_texts.pop(1)
    list_texts.pop(1)
    list_texts.pop(1)
    list_texts.pop(1)
    list_texts.pop(1)
    list_texts.pop(1)
    list_texts.pop(1)
    list_texts.pop(1)
    list_texts.pop(1)
    list_texts.pop(1)
    list_texts.pop(1)
    list_texts.pop(1)
    list_texts.pop(1)
    list_texts.pop(1)

    paragraph_texts = []
    li_list_item_texts = []

    for list_text in list_texts:
        list_item_texts = list_text.find_all("div", class_="ecl")

        list_items_count = len(list_item_texts)
        
        for list_item_text in list_item_texts:
            paragraphs = list_item_text.find_all("p")
            lists = list_item_text.find_all("ul")
            paragraph_text = ""
            li_list_item_text = ""
            
            paragraphs.pop(4)
            paragraphs.pop(4)
            paragraphs.pop(4)
            paragraphs.pop(4)
            paragraphs.pop(4)

            for paragraph in paragraphs:
                paragraph_text = paragraph_text + " " + paragraph.getText()
            
            paragraph_texts.append(paragraph_text)

            lists.pop(1)

            li_list_items = lists[0].find_all("li")

            for li_list_item in li_list_items:
                li_list_item_text = li_list_item_text + ", " + li_list_item.getText()
            
            li_list_item_texts.append(li_list_item_text)


        li_list = ' '.join(li_list_item_texts)
        short_text = ' '.join(paragraph_texts)
        short_text = short_text + li_list

        no_words = len(short_text.split())
        print("SHORT - Number of words: " + str(no_words))

        f =  codecs.open(os.path.abspath(os.curdir) + '/sourcecode/files/' + textLength + '/' + "short_text_file.txt", "w", encoding="utf-16")
        f.write(short_text)
        f.close()
        
        return short_text