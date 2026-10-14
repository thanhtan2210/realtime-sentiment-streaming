"""Build-time helper for spark.Dockerfile.

Resolves the given Maven coordinates (with their dependencies) through
Spark's own Ivy resolver, copies the jars to /opt/spark/jars and writes
/opt/spark/conf/spark-defaults.conf so every Spark session loads them from
disk instead of downloading them at start-up.
"""
import shutil
import sys
from pathlib import Path

import pyspark
from pyspark.sql import SparkSession

JARS_DIR = Path("/opt/spark/jars")
CONF_DIR = Path("/opt/spark/conf")
IVY_DIR = Path("/tmp/ivy")


def main(packages: list[str]) -> None:
    for package in packages:
        if package.startswith("org.apache.spark:") and not package.endswith(f":{pyspark.__version__}"):
            sys.exit(f"{package} does not match pyspark {pyspark.__version__}")

    spark = (
        SparkSession.builder.master("local[1]")
        .config("spark.jars.packages", ",".join(packages))
        .config("spark.jars.ivy", str(IVY_DIR))
        .getOrCreate()
    )
    spark.stop()

    JARS_DIR.mkdir(parents=True, exist_ok=True)
    CONF_DIR.mkdir(parents=True, exist_ok=True)
    jars = sorted(shutil.copy(jar, JARS_DIR) for jar in (IVY_DIR / "jars").glob("*.jar"))
    if not jars:
        sys.exit("no jars were resolved")
    (CONF_DIR / "spark-defaults.conf").write_text(f"spark.jars {','.join(jars)}\n", encoding="utf-8")
    shutil.rmtree(IVY_DIR)
    print(f"Installed {len(jars)} jars in {JARS_DIR}:")
    for jar in jars:
        print(f"  {Path(jar).name}")


if __name__ == "__main__":
    main(sys.argv[1:])
