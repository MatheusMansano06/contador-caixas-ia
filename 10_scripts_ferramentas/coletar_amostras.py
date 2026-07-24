"""Coleta amostras reais da camera ao vivo para retreinar o detector.

Fluxo de uso (com o servidor rodando e a area desenhada):
  1. Recalibre com a area VAZIA no painel.
  2. Negativos (fundo):   python coletar_amostras.py --modo vazio --n 8
  3. Com caixas: coloque N caixas e rode:
       python coletar_amostras.py --modo caixas --esperado 1 --n 10
     A cada captura, MOVA a caixa para outra posicao. O script espera a cena
     estabilizar e so salva quando a quantidade de blocos fisicos bate com
     --esperado (rotulo automatico com controle de qualidade).
  4. Repita com 2, 3 caixas... depois rode o treinar.py.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request

def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.load(r)


def _post(url: str, payload: dict) -> tuple[bool, dict | str]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return True, json.load(r)
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read()).get("detail", str(e))
        except Exception:
            detail = str(e)
        return False, detail


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modo", choices=["vazio", "caixas"], required=True)
    ap.add_argument("--esperado", type=int, default=None,
                    help="Qtd de caixas na cena (controle de qualidade do rotulo)")
    ap.add_argument("--n", type=int, default=10, help="Qtd de amostras a salvar")
    ap.add_argument("--intervalo", type=float, default=4.0,
                    help="Segundos entre capturas (tempo p/ reposicionar)")
    ap.add_argument("--url", default="http://localhost:8000")
    args = ap.parse_args()

    saved = 0
    tries = 0
    max_tries = args.n * 20
    print(f"Coletando {args.n} amostra(s) modo={args.modo}"
          + (f" esperado={args.esperado}" if args.esperado else ""))
    while saved < args.n and tries < max_tries:
        tries += 1
        try:
            status = _get(f"{args.url}/api/status")
        except Exception as e:
            print(f"Servidor indisponivel: {e}")
            return
        if not status.get("running") or not status.get("zone_active"):
            print("Sessao/area inativa. Abra o painel, inicie a camera e desenhe a area.")
            return
        if args.modo == "caixas" and not status.get("stable"):
            time.sleep(1.0)
            continue

        ok, res = _post(f"{args.url}/api/dataset/capture",
                        {"kind": args.modo, "expected": args.esperado})
        if ok:
            saved += 1
            print(f"[{saved}/{args.n}] salvo: {res['saved']} ({res['split']}, "
                  f"{res['caixas_rotuladas']} rotulo(s), total {res['total_amostras']})")
            if saved < args.n and args.modo == "caixas":
                print("  -> Reposicione a(s) caixa(s)...")
            time.sleep(args.intervalo)
        else:
            print(f"  aguardando: {res}")
            time.sleep(2.0)

    print("Coleta encerrada." if saved >= args.n else "Parou sem completar (muitas tentativas).")


if __name__ == "__main__":
    main()
