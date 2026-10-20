# ImagePullBackOff Troubleshooting

## Overview

ImagePullBackOff occurs when Kubernetes cannot pull a container image and delays subsequent attempts.

## Symptoms

- Pod status shows ImagePullBackOff.
- Container fails to start.
- Pod events report image pull errors.
- The pod may initially show ErrImagePull.

## Possible Causes

### Incorrect Image Name or Tag

The image repository or tag may be incorrect, or the requested image may not exist.

### Registry Authentication

The image may be private, and valid registry credentials may be missing or incorrectly configured.

### Registry Connectivity

Cluster nodes may be unable to reach the registry because of DNS, network, firewall, or registry availability issues.

### Registry Rate Limits

The registry may reject image pull requests after a rate limit is reached.

## Diagnostic Commands

Inspect pod status and events:

```bash
kubectl describe pod <pod-name> -n <namespace>