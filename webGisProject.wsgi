import site

site.addsitedir(
    r"C:\Users\zbq\anaconda3\envs\zbq\Lib\site-packages"
)

from run import app
application = app
# import sys
#
# sys.path.insert(0, "F:\\Projects\\Pytorch\\webGisProject\\")


#encoding = 'utf-8'

# def application(environ,start_response):
#     status = "200 Ok"
#     output = b"Hello wsgi"
#     response_headers=[('Content-type','text/plain'),('Content-Length',str(len(output)))]
#     start_response(status,response_headers)
#     return[output]
