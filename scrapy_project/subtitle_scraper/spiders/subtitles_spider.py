import json

import scrapy
import os
import xmlrpc.client as xrpc
import sys

#subtitles_path = "/Users/mesutgurlek/Documents/Machine Learning/project/Movie-Category-Classification-from-Subtitles/Subtitles"
# subtitles_path = "/Users/aeakdogan/hooop/Movie-Category-Classification-from-Subtitles/Subtitles"
subtitles_path = "/home/burak/Documents/Courses-2016f/CS464/Project/Subtitles"
url_template = "http://www.imdb.com/search/title?genres=%s&explore=genres&sort=num_votes,desc&view=simple"
imdb_page_limit = 10

server = xrpc.ServerProxy("http://api.opensubtitles.org/xml-rpc")
# token = server.LogIn("omerakgul58", "omeromer", "en", "2016experimentingwithnlp").get("token")
# token = server.LogIn("randomwalker", "sub1machine", "en", "MachineTitle").get("token")
# token = server.LogIn("gamilgaze", "asdqwe123", "en", "gamil12345").get("token")
token = server.LogIn("alierdogan7", "br12br12", "en", "SubMLProject").get("token")
remaining_quota = server.ServerInfo()['download_limits']['client_download_quota']

print(server.ServerInfo())

categories = ['musical', 'romance', 'horror']
# categories = ['western', 'musical', 'comedy', 'horror', 'war', 'romance', 'adventure', 'action']
subtitle_per_category = 30 #int(remaining_quota / len(categories))

# THIS IS FOR CHECKING IF THE DOWNLOAD LIMIT IS REACHED OR NOT, BEFORE STARTING THE WHOLE DOWNLOADING PROCESS
result = server.DownloadSubtitles(token, ['1953101239']) # an arbitrary subtitle id
if result['data'] == False or result['status'].find('407') > -1:
    print("Download limit reached, not starting the downloading process. Terminating...")
    sys.exit()
else:
    print("Download limit test passed, starting to download...")
#####################################################################


class SubtitlesSpider(scrapy.Spider):
    name = "subtitles"
    # start_urls = ["http://www.opensubtitles.org/en/search/sublanguageid-all/searchonlymovies-on/genre-action/movielanguage-english/movieimdbratingsign-5/movieimdbrating-7/movieyearsign-5/movieyear-1990/offset-0"]

    start_urls = [ url_template % (genre) for genre in categories]

    def parse(self, response):
        # return self.parse_movies(response)

        category_name = response.css(".header ::text").extract_first().split(" ")[2]

        folder_path = "%s/%s" % (subtitles_path, category_name)
        try:
            if not os.path.isdir(folder_path):
                os.mkdir(folder_path, 0o755)
        except OSError:
            print("Directorty cannot be opened in %s, terminating..." % folder_path)
            sys.exit() # for preventing unnecessary quota waste, abort the execution

        #     import shutil
        #     shutil.rmtree(folder_path, ignore_errors=True)
        # finally:
        #     os.mkdir(folder_path, 0o755)

        response.meta['page_limit'] = imdb_page_limit # configure # of pages to be visited to 5
        response.meta['category_name'] = category_name
        return self.parse_imdb_movie_ids(response)

    def parse_imdb_movie_ids(self, response):
        next_page_url = response.urljoin(response.css(".next-page").xpath('@href').extract_first())
        category_name = response.meta['category_name']

        # extract imdb id's from collected links. i.e.  http://www.imdb.com/title/tt0468569/?ref_=adv_li_tt
        # then append those ids to current list of ids
        imdb_ids = response.meta.get('imdb_ids', []) + \
                   [ id.split('/')[2][2:] for id in response.css(".col-title a").xpath("@href").extract() ]

        page_limit = response.meta.get('page_limit', 0) - 1 #decrease the page limit for identifying base case
        if page_limit <= 0:
            return self.parse_movies(imdb_ids=imdb_ids, category_name=category_name)

        yield scrapy.Request(url=next_page_url, callback=self.parse_imdb_movie_ids,
                             meta={'imdb_ids': imdb_ids, 'page_limit': page_limit,
                                   'category_name': category_name})

    def parse_movies(self, imdb_ids, category_name):
        # movie_links = response.css(".bnone").xpath('@href').extract()
        # for link in movie_links:
        #     yield scrapy.Request(url=response.urljoin(link), callback=self.parse_movie)
        print("Listing scraped IMDB ids for%s: ", category_name)
        print(imdb_ids)

        from random import shuffle
        shuffle(imdb_ids) #otherwise it repeatedly tries the first N subtitles.


        subtitles = {} # key => IDSubtitleFile, value => Metadata of subtitle

        # USING IMDB ID'S, DOWNLOAD THEIR METADATA AND SELECT ID OF A SUITABLE SUBTITLE FOR EACH MOVIE
        # for imdb_id in imdb_ids[:subtitle_per_category]:
        remaining = subtitle_per_category
        i = 0
        while remaining > 0 and i < len(imdb_ids):
            imdb_id = imdb_ids[i]
            print("Searching subtitle for movie with ID: %s" % imdb_id)
            found_subtitles = server.SearchSubtitles(token, [{'imdbid': imdb_id, 'sublanguageid': 'eng'}])['data']

            if found_subtitles is None or len(found_subtitles) == 0:
                i += 1
                continue

            print("!!!!! Found %d subtitles for movie with ID: %s " % (len(found_subtitles), imdb_id))

            impaired_subtitles = list(filter(lambda sub: sub['SubHearingImpaired'] == '1' and \
                                                            sub['SubFormat'] == 'srt', found_subtitles))

            impaired_label = ""
            if len(impaired_subtitles) > 0:
                subtitle = impaired_subtitles[0]
                impaired_label = "(IMPAIRED)"
            else:
                # subtitle = found_subtitles[0] # for now get the first subtitle
                # WE DON'T WANT MOVIES WITHOUT IMPAIRED SUPPORT!
                i += 1
                continue

            filename = "%s/%s/%s %s.%s" % (subtitles_path, category_name, subtitle['MovieName'], impaired_label, subtitle['SubFormat'])

            # if this subtitle has already been downloaded before don't append it to array of subtitles to be downloadec
            if not os.path.isfile(filename):
                subtitles[subtitle['IDSubtitleFile']] = {'imdb_id': imdb_id, 'filename': filename,
                                                     'movie_name': subtitle['MovieName'],
                                                    'SubDownloadLink': subtitle['SubDownloadLink']}
                remaining -= 1

            i += 1
            # with open(filename, 'w') as f:
            #     f.write(json.dumps(subtitle['SubDownloadLink']))
            #     # yield parse_movie(imdb_id)


        #DOWNLOAD SUBTITLES AND WRITE THEM INTO FILES
        print("Downloading subtitles of %s" % category_name)
        subtitle_ids = [ idsubtitlefile for idsubtitlefile, sub in subtitles.items()]

        if subtitle_ids == None or len(subtitle_ids) == 0:
            print("No subtitles to download, every found one already exists in Subtitles folder.")
            return None

        # since API allows 20 subtitle downloads at once, divide subtitle_ids into portions 20 by 20 at each iteration
        while len(subtitle_ids) > 0:
            if len(subtitle_ids) > 20:
                portion = subtitle_ids[:20]
                subtitle_ids = subtitle_ids[20:]
            else:
                portion = subtitle_ids
                subtitle_ids = []

            subtitle_files_response = server.DownloadSubtitles(token, portion)

            import base64
            import gzip

            if subtitle_files_response['status'] == '200 OK':
                print("Subtitles downloaded, writing to files...")
                for subtitle_object in subtitle_files_response['data']: #each subtitle_object has base64 data and idsubtitlefile key
                    sub = subtitles[subtitle_object['idsubtitlefile']]

                    # I SOLVED THE ISSUE OPENING THE FILE IN BYTE-WRITING MODE AND DIRECTLY WRITING BYTES OBJECT TO FILE
                    # WITHOUT TRYING TO DECODE THE BYTES INTO A STRING
                    try:
                        with open(sub['filename'], 'wb') as file:
                            file.write(gzip.decompress(base64.b64decode(subtitle_object['data']))) #.decode())
                            print("Subtitle file saved into: %s" % sub['filename'])
                    except FileNotFoundError: #if unsuccessful to open that file, just ignore and skip it
                        continue
            else:
                print("Subtitles cannot be downloaded! Status code: %s" % subtitle_files_response['status'])
                return None