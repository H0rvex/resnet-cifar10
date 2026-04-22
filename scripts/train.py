"""CLI entry: train ResNet on CIFAR-10 (orchestration lives in ``resnet_cifar10.train``)."""

from resnet_cifar10.train import build_parser, resolve_config, train


def main() -> None:
    args = build_parser().parse_args()
    cfg = resolve_config(args)
    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
