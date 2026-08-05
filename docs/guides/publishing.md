# Publishing the docs

SATARK docs are built with MkDocs Material into a static `site/` folder. You can host that site on **GitHub Pages** or any **external domain / static host**.

## Option A — GitHub Pages (recommended)

After this workflow is on `main`, do a one-time repo setup:

1. Open the repo on GitHub → **Settings** → **Pages**
2. Under **Build and deployment** → **Source**, choose **GitHub Actions**
3. Merge/push to `main` (or run the **Docs** workflow via **Actions** → **Docs** → **Run workflow**)

Your site URL will be:

```text
https://kaushalbhavsar.github.io/satark/
```

That matches `site_url` in `mkdocs.yml`.

### What the workflow does

- On pull requests: builds docs with `mkdocs build --strict` (no deploy)
- On push to `main`: builds, uploads the `site/` artifact, deploys to GitHub Pages

### Manual local preview

```bash
pip install -e ".[docs]"
mkdocs serve
```

### Manual local build

```bash
mkdocs build --strict
# output: ./site
```

## Option B — Custom domain on GitHub Pages

Use this if you own a domain (for example `docs.satark.dev` or `satark.example.com`).

1. Keep GitHub Pages enabled (Option A)
2. Add a `CNAME` file in the published site root with your domain:
   - Easiest with MkDocs: set in `mkdocs.yml`:

```yaml
site_url: https://docs.example.com/
extra:
  # optional
```

And create `docs/CNAME` (MkDocs copies root-level files from `docs/`):

```text
docs.example.com
```

3. In GitHub → **Settings** → **Pages** → **Custom domain**, enter the same hostname and save
4. At your DNS provider, add one of:

| Goal | DNS record |
|------|------------|
| Apex domain (`example.com`) | `A` records to GitHub Pages IPs, or `ALIAS`/`ANAME` if supported |
| Subdomain (`docs.example.com`) | `CNAME` → `kaushalbhavsar.github.io` |

5. Wait for DNS + TLS. GitHub can issue HTTPS automatically once DNS verifies.

Update `site_url` in `mkdocs.yml` to the custom domain so links and SEO stay correct.

## Option C — External host (Netlify, Cloudflare Pages, Vercel, S3, …)

Any static host works because MkDocs emits plain HTML/CSS/JS.

### Build command

```bash
pip install -e ".[docs]"
mkdocs build --strict
```

### Publish directory

```text
site
```

### Netlify example

- Build command: `pip install -e ".[docs]" && mkdocs build --strict`
- Publish directory: `site`
- Add a custom domain in the Netlify UI and follow their DNS instructions

### Cloudflare Pages example

- Framework preset: None
- Build command: `pip install -e ".[docs]" && mkdocs build --strict`
- Output directory: `site`
- Attach your domain in Cloudflare Pages → Custom domains

### Vercel example

- Build command: `pip install -e ".[docs]" && mkdocs build --strict`
- Output directory: `site`

### Generic nginx / VPS

```bash
mkdocs build --strict
rsync -av --delete site/ user@server:/var/www/satark-docs/
```

Point your domain’s DNS `A`/`CNAME` at that server and configure TLS (e.g. Caddy or certbot).

## Checklist

- [ ] Docs build locally with `mkdocs build --strict`
- [ ] GitHub Pages source set to **GitHub Actions** (for Option A/B)
- [ ] `site_url` in `mkdocs.yml` matches the public URL
- [ ] For custom domains: DNS + `CNAME` / Pages custom domain configured
- [ ] HTTPS works before sharing the link
