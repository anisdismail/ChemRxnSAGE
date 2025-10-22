# Contributing to ChemRxnSAGE

First off, thank you for considering contributing to ChemRxnSAGE! It's people like you that make open source such a great community.

## Where do I go from here?

If you've noticed a bug or have a feature request, [make one](https://github.com/anisdismail/ChemRxnSAGE/issues/new)! It's generally best if you get confirmation of your bug or approval for your feature request this way before starting to code.

### Fork & create a branch

If this is something you think you can fix, then [fork ChemRxnSAGE](https://github.com/anisdismail/ChemRxnSAGE/fork) and create a branch with a descriptive name.

A good branch name would be (where issue #33 is the ticket you're working on):

```sh
git checkout -b 33-add-new-model
```

### Get the test suite running

Make sure you're running the test suite locally before you start making any changes.

### Implement your fix or feature

At this point, you're ready to make your changes! Feel free to ask for help; everyone is a beginner at first :smile_cat:

### Make a Pull Request

At this point, you should switch back to your master branch and make sure it's up to date with ChemRxnSAGE's master branch:

```sh
git remote add upstream git@github.com:anisdismail/ChemRxnSAGE.git
git checkout master
git pull upstream master
```

Then update your feature branch from your local copy of master, and push it!

```sh
git checkout 33-add-new-model
git rebase master
git push --force origin 33-add-new-model
```

Finally, go to GitHub and [make a Pull Request](https://github.com/anisdismail/ChemRxnSAGE/compare)

### Keeping your Pull Request updated

If a maintainer asks you to "rebase" your PR, they're saying that a lot of code has changed, and that you need to update your branch so it's easier to merge.

To learn more about rebasing, check out this guide on [dev.to](https://dev.to/stsewd/github-rebase-and-merge-4663).

## How to get in touch

You can reach out to us on [GitHub Issues](https://github.com/anisdismail/ChemRxnSAGE/issues).

## Code of Conduct

Please note that this project is released with a Contributor Code of Conduct. By participating in this project you agree to abide by its terms. You can find the Code of Conduct in the `CODE_OF_CONDUCT.md` file.
