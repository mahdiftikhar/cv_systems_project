"""List installed distribution versions without requiring pip or uv at runtime.

This is an environment inventory, not a dependency lockfile.
"""

from importlib.metadata import distributions


def main():
    packages = sorted(
        f"{package.metadata['Name']}=={package.version}"
        for package in distributions()
    )
    print("\n".join(packages))


if __name__ == "__main__":
    main()
