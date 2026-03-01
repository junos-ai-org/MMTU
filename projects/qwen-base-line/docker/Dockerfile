FROM mmtu-qwen7b:latest

WORKDIR /workspace/MMTU
COPY . /workspace/MMTU

COPY projects/qwen-base-line/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
