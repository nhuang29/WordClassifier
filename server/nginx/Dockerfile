# Pull nginx from the Docker hub
FROM nginx

# Remove the default nginx config file and replace
RUN rm /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/conf.d/
