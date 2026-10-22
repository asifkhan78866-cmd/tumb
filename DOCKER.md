# Running this on someone else's machine, with Docker

Docker is the way out of the Python/torch/architecture problems: the image
carries Python 3.11, a CPU build of PyTorch, the four trained checkpoints, the
metrics they were measured with, and a few sample images. The receiving machine
needs **Docker Desktop and nothing else** — no Python, no Homebrew, no dataset.

> **Not verified on this machine.** Docker is not installed here, so these files
> were written and reviewed but never built. Expect to fix something small on
> the first run; the commands below say what to look for.

---

## Which route to use

| | the friend's machine needs | you need | good when |
|---|---|---|---|
| **A. Clone and build** | Docker + git | nothing | they are comfortable with a terminal |
| **B. Registry** | Docker + an account | a GHCR/Docker Hub login | you will send updates more than once |
| **C. A file** | Docker | ~1.5 GB of transfer | no accounts, no git, one-off |

---

## A. They clone and build it (simplest)

```bash
git clone https://github.com/asifkhan78866-cmd/tumb.git
cd tumb
docker compose up --build
```

Then: frontend <http://localhost:3000>, API docs <http://localhost:8000/docs>.

The first build takes 10–20 minutes (PyTorch is large). The backend prints which
methods it can serve as it starts:

```
[image] 1. Pre-trained CNNs + Transfer Learning: ready
[image] 2. Red Fox Optimized ZFNet: ready
[image] 3. U-Net + ConvLSTM + SFLA: ready
[image] 4. MRI–SPECT Fusion: ready
```

If any line says `NO CHECKPOINT`, the weights did not reach the image — check
that `.dockerignore` still has no `backend/weights` rule in it.

---

## B. Push to a registry, they pull

**The architecture trap:** an image built on an Apple Silicon Mac is `arm64` and
will not start on an ordinary Windows/Linux PC. Build for their CPU explicitly.

```bash
# once: a GitHub personal access token with write:packages
echo "$GITHUB_TOKEN" | docker login ghcr.io -u asifkhan78866-cmd --password-stdin

# build for a normal PC (amd64) and push
docker buildx build --platform linux/amd64 \
  -f backend/Dockerfile -t ghcr.io/asifkhan78866-cmd/neuroseg-backend:latest --push .

docker buildx build --platform linux/amd64 \
  -f frontend/Dockerfile --build-arg NEXT_PUBLIC_API_URL=http://localhost:8000 \
  -t ghcr.io/asifkhan78866-cmd/neuroseg-frontend:latest --push .
```

Build for both CPU families at once with
`--platform linux/amd64,linux/arm64`; it takes about twice as long.

On their machine (no git, no clone — this is the whole thing):

```bash
docker run -d -p 8000:8000 --name neuroseg-backend ghcr.io/asifkhan78866-cmd/neuroseg-backend:latest
docker run -d -p 3000:3000 --name neuroseg-frontend ghcr.io/asifkhan78866-cmd/neuroseg-frontend:latest
```

Packages published to GHCR are private by default — make them public on the
package's page, or have them `docker login` first.

---

## C. Send them a file

No accounts. Build, save, transfer, load.

```bash
# build for THEIR cpu, not yours
docker buildx build --platform linux/amd64 -f backend/Dockerfile -t neuroseg-backend:latest --load .
docker buildx build --platform linux/amd64 -f frontend/Dockerfile \
  --build-arg NEXT_PUBLIC_API_URL=http://localhost:8000 -t neuroseg-frontend:latest --load .

# one file with both images, compressed
docker save neuroseg-backend:latest neuroseg-frontend:latest | gzip > neuroseg-images.tar.gz
```

Expect roughly 1.5–2 GB. Send it however you like (Drive, WeTransfer, a USB
stick), then on their machine:

```bash
docker load < neuroseg-images.tar.gz
docker run -d -p 8000:8000 --name neuroseg-backend neuroseg-backend:latest
docker run -d -p 3000:3000 --name neuroseg-frontend neuroseg-frontend:latest
```

They will also want `docker-compose.yml` if they prefer `docker compose up`
(it refers to the same image names).

---

## The terminal runner, inside the container

Every method on one image, with its pictures and explanations:

```bash
docker exec -it neuroseg-backend \
  python -m backend.cli.run_methods data/bri/archive/Testing/meningioma/Te-me_0014.jpg
```

To run it on *their* image rather than the bundled samples, mount a folder:

```bash
docker run --rm -v "$PWD/scans:/scans" neuroseg-backend:latest \
  python -m backend.cli.run_methods /scans/their-mri.png --images blocks
```

---

## What the image does and does not contain

**In:** the code, the four trained checkpoints, both SVM bundles, both U-Nets,
every metrics file, run card and plot, and nine sample images.

**Out:** the datasets (gigabytes — see [data/README.md](data/README.md)), the
notebooks, the caches, `.env`.

So predictions work immediately; retraining inside the container needs a dataset
mounted in (`docker-compose.yml` has the line, commented out).

---

## When something goes wrong

| symptom | cause | fix |
|---|---|---|
| `exec format error` on start-up | the image was built for a different CPU | rebuild with `--platform linux/amd64` |
| frontend loads, every request fails | `NEXT_PUBLIC_API_URL` is compiled into the bundle at build time | rebuild the frontend with the address *their browser* will use |
| `NO CHECKPOINT` at start-up | the weights were excluded from the build context | check `.dockerignore` |
| build killed around torch | Docker Desktop's memory limit | raise it to 8 GB in Settings → Resources |
| port already in use | something else holds 3000/8000 | `-p 3001:3000`, and rebuild the frontend if you move 8000 |
