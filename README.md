# Uruchomienie

Konieczne jest zainstalowanie i uruchomienie Docker Desktop.

Następnie, aby uruchomić aplikację w kontenerze dockerowym, z poziomu projektu należy wykonać komendę:

```{text}
docker build -t streamlit-dashboard .
```

a gdy skończy się budowanie:

```{text}
docker run -p 8502:8502 --name social-media-dashboard streamlit-dashboard
```

Po pełnym załadowaniu, aplikacja będzie dostępna pod adresem http://localhost:8502/

# Start-up

It's necessary to have installed and started the Docker Desktop app.

Then, to launch the application in your docker container, you must first run the following commands from the project directory:

```{text}
docker build -t streamlit-dashboard .
```

and when the build is complete:

```{text}
docker run -p 8502:8502 --name social-media-dashboard streamlit-dashboard
```

After some time, all the modules will have loaded and the app will be available at http://localhost:8502/
