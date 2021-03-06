from tornado.httpserver import HTTPServer
from tornado.wsgi import WSGIContainer
from run import app
from tornado.ioloop import IOLoop

s = HTTPServer(WSGIContainer(app))
s.listen(9900)
IOLoop.current().start()