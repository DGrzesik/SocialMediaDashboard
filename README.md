# Uruchomienie

Aby uruchomić aplikację w kontenerze dockerowym, należy wykonać komendę:

```{text}
docker build -t streamlit-dashboard .
```

a następnie, gdy skończy się budowanie:

```{text}
docker run -p 8501:8501 --name social-media-dashboard streamlit-dashboard
```

Po pełnym załadowaniu, aplikacja będzie dostępna pod adresem http://localhost:8501/

# Start-up

To launch the application in your docker container, you must first run the following commands:

```{text}
docker build -t streamlit-dashboard .
```

and when the build is complete:

```{text}
docker run -p 8501:8501 --name social-media-dashboard streamlit-dashboard
```

After some time, all the modules will have loaded and the app will be available at http://localhost:8501/