# Tracking Data Remotely with DVC (Google Drive)

Install the Google Drive dependencies for DVC using

```bash
pip install 'dvc[gdrive]'  
```

Add the Google Drive remote using the unique identifier found in the URL of your Drive folder:

```bash
dvc remote add driveremote gdrive://UNIQUE_IDENTIFIER
```

create a file and push using

```bash
dvc push --remote driveremote
```

or you can now set the Google Drive remote as your default:
```bash
dvc remote default newremote
dvc push
```
