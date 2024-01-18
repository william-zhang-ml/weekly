"""Code for easily-creating HTML reports w/a pre-defined figure style. """
import os
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
import yaml


_MY_DIR = Path(__file__).parent
with open(_MY_DIR / 'mpl_style.yaml', mode='r', encoding='utf-8') as file:
    MPL_STYLE = yaml.safe_load(file)


class Report:
    """Utility class to build HTML reports with figures and metric tables. """
    def __init__(self, report_path: str) -> None:
        """
        Args:
            report_path (str): directory in which to write report and images
        """
        self._report_path = Path(report_path)
        os.makedirs(report_path)
        self._figures = []
        self._metrics = {}

    def register_image_path(self, imgname: str) -> str:
        """Internally track a new image to add to the report.

        Args:
            imgname (str): image filepath (foo.jpg for example)

        Returns:
            str: full filepath of where to save the image
        """
        self._figures.append(imgname)
        return str(self._report_path / imgname)

    def log_metric(self, name: str, value: float, fmt: str = '.03f') -> None:
        """Internally track a new metric to add to the report.

        Args:
            name (str): name of the metric (F1-score for example)
            value (float): metric value to report
            fmt (str): f-string format for <value>
        """
        self._metrics.update({name: f'{float(value): {fmt}}'})

    def fill_template(self) -> str:
        """Put tracked image paths & metrics into report template. """
        environment = Environment(loader=FileSystemLoader(_MY_DIR))
        template = environment.get_template('template.html')
        content = template.render(
            figures=self._figures,
            metrics=self._metrics
        )
        return content

    def write(self) -> None:
        """Compile report and write to disk. """
        path = self._report_path / 'report.html'
        with open(path, mode='w', encoding='utf-8') as report_file:
            report_file.write(self.fill_template())
