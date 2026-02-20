# This file is part of the Flask web framework and is licensed under the BSD license.
# See the LICENSE file in the root directory for more information.

"""
    flask.app
    ~~~~~~~~~

    This module implements the central WSGI application object.

    :copyright: © 2010 by the Pallets team.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import os
import sys
import typing as t
from datetime import timedelta
from functools import update_wrapper
from threading import Lock

from werkzeug.datastructures import Headers
from werkzeug.exceptions import HTTPException, NotFound as NotFound
from werkzeug.routing import BuildError, MapAdapter
from werkzeug.serving import WSGIRequestHandler
from werkzeug.serving import make_server as werkzeug_make_server
from werkzeug.test import Client
from werkzeug.wrappers import Request as Request
from werkzeug.wrappers import Response as Response

from . import cli
from . import json
from .config import Config
from .ctx import AppContext
from .ctx import RequestContext
from .globals import _app_ctx_stack
from .globals import _request_ctx_stack
from .helpers import _endpoint_from_view_func
from .helpers import get_debug_flag
from .helpers import get_env
from .helpers import get_flashed_messages
from .helpers import url_for
from .sessions import SecureCookieSessionInterface
from .sessions import Session
from .signals import appcontext_tearing_down
from .signals import got_request_exception
from .signals import request_finished
from .signals import request_started
from .signals import request_tearing_down
from .templating import DispatchingJinjaLoader
from .templating import Environment
from .typing import AfterRequestCallable
from .typing import AppOrBlueprintKey
from .typing import BeforeFirstRequestCallable
from .typing import BeforeRequestCallable
from .typing import ErrorHandlerCallable
from .typing import TeardownCallable
from .typing import TemplateContextProcessorCallable
from .typing import URLDefaultCallable
from .typing import URLValuePreprocessorCallable
from .wrappers import Request as Request
from .wrappers import Response as Response

if t.TYPE_CHECKING:
    from .testing import FlaskClient


def _make_timedelta(value: timedelta | int | None) -> timedelta | None:
    """Convert a value to a :class:`datetime.timedelta`.

    :param value: The value to convert.
    """
    if value is None:
        return None

    if isinstance(value, timedelta):
        return value

    return timedelta(seconds=value)


class Flask:
    """The flask object implements a WSGI application and acts as the central
    object.  It is passed the name of the module or package of the
    application.  Once it is created it will act as a central registry for
    the view functions, the URL rules, template configuration and much more.

    The name of the package is used to resolve resources from inside the
    package or the folder the module is contained in depending on if the
    package parameter resolves to an actual python package (a folder with
    an :file:`__init__.py` file inside) or a standard module (just a ``.py`` file).

    For more information about resource loading, see :func:`open_resource`.

    Usually you create a :class:`Flask` instance in your main module or
    in the :file:`__init__.py` file of your package like this::

        from flask import Flask
        app = Flask(__name__)

    .. admonition:: About the First Parameter

        The idea of the first parameter is to give Flask an idea of what
        belongs to your application.  This name is used to find resources
        on the filesystem, can be used by extensions to improve debugging
        information and a lot more.

        So it's important what you provide there.  If you are using a single
        module, ``__name__`` is always the correct value.  If you however are
        using a package, it's usually recommended to hardcode the name of
        your package there.

        For example if your application is defined in ``yourapplication/app.py``
        you should create it with one of the two versions below::

            app = Flask('yourapplication')
            app = Flask(__name__.split('.')[0])

    Why is that?  The application will work even with ``__name__``, thanks
    to how resources are looked up.  However it will make debugging more
    painful.  Certain extensions can make assumptions based on the import
    name of your application.

    .. versionchanged:: 1.0
        The ``static_url_path`` parameter was removed.

    .. versionchanged:: 0.8
        Support for ``instance_relative_config`` was added.

    .. versionchanged:: 0.7
        The ``static_folder`` parameter was added.

    :param import_name: the name of the application package
    :param static_url_path: can be used to specify a different path for the
                            static files on the web.  Defaults to the name
                            of the ``static_folder`` folder.
    :param static_folder: the folder with static files that should be served
                          at ``static_url_path``.  Defaults to the ``'static'``
                          folder in the root path of the application.
    :param static_host: the host to use when adding the static route.
                        Defaults to None.
    :param host_matching: set ``True`` if the application should use the
                          host matching feature of Werkzeug.
    :param subdomain_matching: set ``True`` if the application should use
                               the subdomain matching feature of Werkzeug.
    :param template_folder: the folder that contains the templates that should
                            be used by the application.  Defaults to
                            ``'templates'`` folder in the root path of the
                            application.
    :param instance_path: An alternative instance path for the application.
                           By default the folder ``'instance'`` next to the
                           package or module is assumed to be the instance
                           path.
    :param instance_relative_config: if set to ``True`` relative configuration
                                      filenames are assumed to be relative to
                                      the instance path instead of the
                                      application root.
    :param root_path: The path to the root of the application files.
    """

    #: The class that is used for request objects.  See :class:`~flask.wrappers.Request`
    #: for more information.
    request_class = Request

    #: The class that is used for response objects.  See :class:`~flask.wrappers.Response`
    #: for more information.
    response_class = Response

    #: The class that is used for the Jinja environment.
    #: See :class:`~flask.templating.Environment`.
    jinja_environment = Environment

    #: The class that is used for the :data:`~flask.g` object.
    #: See :class:`~flask.ctx.AppContext`.
    app_ctx_globals_class = AppContext

    #: The class that is used for the configuration object.
    #: See :class:`~flask.config.Config`.
    config_class = Config

    #: The class that is used for the test client.
    #: See :class:`~flask.testing.FlaskClient`.
    test_client_class: t.Type[FlaskClient] | None = None

    #: The class that is used for the session interface.
    #: See :class:`~flask.sessions.SessionInterface`.
    session_interface = SecureCookieSessionInterface()

    def __init__(
        self,
        import_name: str,
        static_url_path: str | None = None,
        static_folder: str | None = "static",
        static_host: str | None = None,
        host_matching: bool = False,
        subdomain_matching: bool = False,
        template_folder: str | None = "templates",
        instance_path: str | None = None,
        instance_relative_config: bool = False,
        root_path: str | None = None,
    ) -> None:
        self._got_first_request = False
        self._before_request_lock = Lock()
        self._before_request_funcs: dict[AppOrBlueprintKey, list[BeforeRequestCallable]] = {}
        self._after_request_funcs: dict[AppOrBlueprintKey, list[AfterRequestCallable]] = {}
        self._teardown_request_funcs: dict[
            AppOrBlueprintKey, list[TeardownCallable]
        ] = {}
        self._teardown_appcontext_funcs: dict[
            AppOrBlueprintKey, list[TeardownCallable]
        ] = {}
        self._url_default_functions: dict[
            AppOrBlueprintKey, list[URLDefaultCallable]
        ] = {}
        self._url_value_preprocessors: dict[
            AppOrBlueprintKey, list[URLValuePreprocessorCallable]
        ] = {}
        self._template_context_processors: dict[
            AppOrBlueprintKey, list[TemplateContextProcessorCallable]
        ] = {}
        self._error_handlers: dict[
            AppOrBlueprintKey, dict[t.Type[Exception], ErrorHandlerCallable]
        ] = {}
        self._before_first_request_funcs: list[BeforeFirstRequestCallable] = []

        self.name = import_name
        self.url_map = self.url_map_class()
        self.static_folder = static_folder
        self.static_url_path = static_url_path
        self.static_host = static_host
        self.host_matching = host_matching
        self.subdomain_matching = subdomain_matching
        self.template_folder = template_folder
        self.instance_path = instance_path
        self.instance_relative_config = instance_relative_config
        self.root_path = root_path

        self.config = self.config_class(
            self.root_path, self.default_config, self.instance_relative_config
        )

        self.view_functions: dict[str, t.Callable] = {}
        self.blueprints: dict[str, "Blueprint"] = {}
        self.extensions: dict[str, t.Any] = {}
        self.shell_context_processors: list[t.Callable[[], dict[str, t.Any]]] = []

        self._cli_commands = cli.AppGroup()

        self._register_error_handler(NotFound, self.handle_user_exception)
        self._register_error_handler(HTTPException, self.handle_http_exception)

    def _register_error_handler(
        self,
        exception_class: type[Exception],
        handler: ErrorHandlerCallable,
    ) -> None:
        """Register an error handler for the given exception class."""
        self._error_handlers.setdefault(None, {})[exception_class] = handler

    @property
    def name(self) -> str:
        """The name of the application.  This is usually the import name
        with the difference that it's guessed from the run file if the
        import name is "__main__".
        """
        if self._name == "__main__":
            base = os.path.basename(sys.argv[0])
            if base.endswith(".py"):
                base = base[:-3]
            return base
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.name!r}>"
